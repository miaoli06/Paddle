# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from _collections import defaultdict

import paddle
import paddle.fluid.framework as framework
from paddle.distributed.passes.pass_base import PassBase, register_pass
from paddle.framework import core
from paddle.static import Parameter, Program
from ..ps.utils.public import *  # noqa: F403
from paddle.distributed.fleet.meta_optimizers.common import (
    OpRole
)

@register_pass("auto_adjust_op")
class AutoAdjustOpPass(PassBase):
    def __init__(self):
        super().__init__()
        self._op_num = 0
        self._op_up_sums = None
        self._op_down_sums = None
        self._op_down_idxs = None
        self._op_up_idxs = None
        self._op_weights = None
        self._op_order_idx = []
        self._op_unused_idx = []
        self._op_types = []

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True
    
    def _get_input_indexes(self, op, other_op, used_input_names):
        in_count = 0
        other_output_names = other_op.desc.output_arg_names()
        for name in op.desc.input_arg_names():
            if name in used_input_names:
                continue
            if not name in other_output_names:
                continue
            used_input_names.add(name)
            in_count = 1
        return in_count
        
    def _get_input_ops(self):
        ops_idx = []
        for i in range(self._op_num):
            if self._op_up_sums[i] > 0:
                continue
            if self._op_down_sums[i] == 0:
                continue
            ops_idx.append(i)
        return ops_idx
    
    def _get_op_weight(self, idx):
        if not idx in self._op_down_idxs:
            return 0
        stack_idxs = []
        for id in self._op_down_idxs[idx]:
            stack_idxs.append(id)
            
        ref_count = 0
        used_idx_set = set()
        while len(stack_idxs) > 0:
            id = stack_idxs.pop()
            if id in used_idx_set:
                continue
            used_idx_set.add(id)
            ref_count = ref_count + 1
            # not in down idxs
            if not id in self._op_down_idxs:
                continue
            for child_id in self._op_down_idxs[id]:
                stack_idxs.append(child_id)
            
        return ref_count
    
    def _get_op_role_id(self, op):
        if "op_role" in op.attr_names:
            return int(op.attr('op_role'))
        return 0
    
    def _build_op_weights(self, ops):
        self._op_weights = [None] * self._op_num
        for idx in range(self._op_num):
            used = False
            weight = 0
            if self._op_up_sums[idx] == 0 and \
                self._op_down_sums[idx] == 0:
                self._op_unused_idx.append(idx)
                used = True
            else:
                weight = self._get_op_weight(idx)
            self._op_weights[idx] = {
                "idx" : idx, 
                "weight" : weight, 
                "used" : used,
                "name" : self._op_types[idx],
                "op_role": self._get_op_role_id(ops[idx]),
            }

    def _build_op_depends(self, program):
        block = program.global_block()
        self._op_num = len(block.ops)
        
        self._op_up_sums = [0] * self._op_num
        self._op_down_sums = [0] * self._op_num
        self._op_down_idxs = {}
        self._op_up_idxs = {}
        
        #for op_idx, op in reversed(list(enumerate(block.ops))):
        for op_id, op in enumerate(block.ops):
            if op_id == 0:
                self._op_types.append(op.type)
                continue
            in_sum = 0
            args_in_num = len(op.desc.input_arg_names())
            used_input_names = set()
            max_id = op_id - 1
            pre_idxs = []
            for pre_op_id in range(max_id, -1, -1):
                in_count = self._get_input_indexes(op, block.ops[pre_op_id], used_input_names)
                if in_count > 0:
                    if not pre_op_id in self._op_down_idxs:
                        self._op_down_idxs[pre_op_id] = []
                    self._op_down_idxs[pre_op_id].append(op_id)
                    self._op_down_sums[pre_op_id] = self._op_down_sums[pre_op_id] + 1
                    pre_idxs.append(pre_op_id)
                in_sum = in_sum + in_count
            self._op_up_sums[op_id] = in_sum
            self._op_up_idxs[op_id] = pre_idxs
            self._op_types.append(op.type)
            
        #print("_op_types=", self._op_types)
        #print("_op_up_sums=", self._op_up_sums)
        #print("_op_down_sums=", self._op_down_sums)
        #print("_op_down_idxs=", self._op_down_idxs)
        # build op weights
        self._build_op_weights(block.ops)
        #print("_op_weights=", self._op_weights)
        
    def _get_next_idx(self):
        max_weight = 0
        cur_idx = -1
        min_role_id = 5
        for idx in range(self._op_num):
            if self._op_weights[idx]["used"]:
                continue
            cur_weight = self._op_weights[idx]["weight"]
            op_role = self._op_weights[idx]["op_role"]
            if max_weight > cur_weight or min_role_id < op_role:
                continue
            cur_idx = idx
            max_weight = cur_weight
            min_role_id = op_role
        return cur_idx
    
    def _is_optimize_op(self, op):
        if "op_role" in op.attr_names and (
            int(op.attr('op_role')) == int(OpRole.Optimize)):
            return True
        return False
    
    def _is_unused_op(self, op, block):
        if "is_sparse" in op.attr_names and op.attr('is_sparse'):
            return False
        if op.type == "py_func":
            return False
        for name in op.desc.output_arg_names():
            var = block.var(name)
            if not var.persistable:
                continue
            return False
        for name in op.desc.input_arg_names():
            var = block.var(name)
            if not var.persistable:
                continue
            return False
        return True
    
    def _reset_block_op_order(self, program, op_order_idxs):
        block = program.global_block()
        for idx in op_order_idxs:
            op = block.ops[idx]
            tmp_op_desc = block.desc.append_op()
            tmp_op_desc.copy_from(op.desc)
        for _ in range(self._op_num):
            block._remove_op(0, sync=False)
        block.ops.clear()
        block._sync_with_cpp()
            
    def _adjust_op_order(self, program):
        self._op_order_idx = []
        idxs = self._get_input_ops()
        for idx in idxs:
            self._op_order_idx.append(idx)
            self._op_weights[idx]["used"] = True
        
        while True:
            idx = self._get_next_idx()
            if idx < 0:
                break
            self._op_weights[idx]["used"] = True
            self._op_order_idx.append(idx)
            
        block = program.global_block()
        new_types = []
        new_op_order_idx = []
        for i in self._op_order_idx:
            op = block.ops[i]
            if self._is_optimize_op(op):
                new_op_order_idx.append(i)
                new_types.append([i, op.type])
                continue
            if self._op_down_sums[i] == 0:
                if self._is_unused_op(op, block):
                    self._op_unused_idx.append(i)
                else:
                    new_op_order_idx.append(i)
                    new_types.append([i, op.type])
            else:
                new_op_order_idx.append(i)
                new_types.append([i, op.type])
        
        for i in self._op_unused_idx:
            print("unused idx=%s, %s" % (i, block.ops[i]))
        
        self._op_order_idx = new_op_order_idx
        print("adjust new order = ", self._op_order_idx)
        print("new op types = ", new_types)
        self._reset_block_op_order(program, self._op_order_idx)

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        self._build_op_depends(main_program)
        self._adjust_op_order(main_program)