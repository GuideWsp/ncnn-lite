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

#ifndef LAYER_RESHAPE_H
#define LAYER_RESHAPE_H

#include "layer.h"

struct Reshape
{
    // layer base
    Layer layer;

    // proprietary data
    // reshape flag
    // 0 = copy from bottom
    // -1 = remaining
    // -233 = drop this dim (default)
    int w;
    int h;
    int c;
    int permute;
    int ndim;
};

void *Reshape_ctor(void *_self, va_list *args);

int Reshape_load_param(void *_self, const ParamDict& pd);

int Reshape_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

// default operators
#define Reshape_dtor                     Layer_dtor
#define Reshape_load_model               Layer_load_model
#define Reshape_create_pipeline          Layer_create_pipeline
#define Reshape_destroy_pipeline         Layer_destroy_pipeline
#define Reshape_forward_multi            Layer_forward_multi
#define Reshape_forward_inplace_multi    Layer_forward_inplace_multi
#define Reshape_forward_inplace          Layer_forward_inplace

#endif // LAYER_RESHAPE_H
