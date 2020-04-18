// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_HARDSWISH_H
#define LAYER_HARDSWISH_H

#include "layer.h"

struct HardSwish
{
    // layer base
    Layer layer;

    // proprietary data
    float alpha, beta, lower, upper;
};

void *HardSwish_ctor(void *_self, va_list *args);

int HardSwish_load_param(void *_self, const ParamDict& pd);

int HardSwish_forward_inplace(void *_self, Mat& bottom_top_blob, const Option& opt);

#define HardSwish_dtor                     Layer_dtor
#define HardSwish_load_model               Layer_load_model
#define HardSwish_create_pipeline          Layer_create_pipeline
#define HardSwish_destroy_pipeline         Layer_destroy_pipeline
#define HardSwish_forward_multi            Layer_forward_multi
#define HardSwish_forward                  Layer_forward
#define HardSwish_forward_inplace_multi    Layer_forward_inplace_multi

#endif // LAYER_HARDSWISH_H
