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

#ifndef LAYER_POOLING_H
#define LAYER_POOLING_H

#include "layer.h"

struct Pooling
{
    // layer base
    Layer layer;

    // proprietary data
    // param
    int pooling_type;
    int kernel_w;
    int kernel_h;
    int stride_w;
    int stride_h;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    int global_pooling;
    int pad_mode;// 0=full 1=valid 2=SAME_UPPER 3=SAME_LOWER
    int avgpool_count_include_pad;
};

enum PoolMethod { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };

void *Pooling_ctor(void *_self, va_list *args);

int Pooling_load_param(void *_self, const ParamDict& pd);

int Pooling_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

void Pooling_make_padding(void *_self, const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt);

// default operators
#define Pooling_dtor                     Layer_dtor
#define Pooling_load_model               Layer_load_model
#define Pooling_create_pipeline          Layer_create_pipeline
#define Pooling_destroy_pipeline         Layer_destroy_pipeline
#define Pooling_forward_multi            Layer_forward_multi
#define Pooling_forward_inplace_multi    Layer_forward_inplace_multi
#define Pooling_forward_inplace          Layer_forward_inplace

#endif // LAYER_POOLING_H
