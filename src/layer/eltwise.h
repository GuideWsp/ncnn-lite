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

#ifndef LAYER_ELTWISE_H
#define LAYER_ELTWISE_H

#include "layer.h"

struct Eltwise
{
    // layer base
    Layer layer;

    // proprietary data
    // param
    int op_type;
    Mat coeffs;
};

enum OperationType { Operation_PROD = 0, Operation_SUM = 1, Operation_MAX = 2 };

void *Eltwise_ctor(void *_self, va_list *args);

int Eltwise_load_param(void *_self, const ParamDict& pd);

int Eltwise_forward(void *_self, const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt);

// default operators
#define Eltwise_dtor                     Layer_dtor
#define Eltwise_load_model               Layer_load_model
#define Eltwise_create_pipeline          Layer_create_pipeline
#define Eltwise_destroy_pipeline         Layer_destroy_pipeline
#define Eltwise_forward_multi            Layer_forward_multi
#define Eltwise_forward_inplace_multi    Layer_forward_inplace_multi
#define Eltwise_forward_inplace          Layer_forward_inplace

#endif // LAYER_ELTWISE_H
