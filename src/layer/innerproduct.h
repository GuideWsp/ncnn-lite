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

#ifndef LAYER_INNERPRODUCT_H
#define LAYER_INNERPRODUCT_H

#include "layer.h"

struct InnerProduct
{
    // layer base
    Layer layer;

    // proprietary data
    // param
    int num_output;
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    // model
    Mat weight_data;
    Mat bias_data;

    Mat weight_data_int8_scales;
    float bottom_blob_int8_scale;
};

void *InnerProduct_ctor(void *_self, va_list *args);

int InnerProduct_load_param(void *_self, const ParamDict& pd);

int InnerProduct_load_model(void *_self, const ModelBin& mb);

int InnerProduct_create_pipeline(void *_self, const Option& opt);

int InnerProduct_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

int InnerProduct_forward_int8(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

// default operators
#define InnerProduct_dtor                     Layer_dtor
#define InnerProduct_destroy_pipeline         Layer_destroy_pipeline
#define InnerProduct_forward_multi            Layer_forward_multi
#define InnerProduct_forward_inplace_multi    Layer_forward_inplace_multi
#define InnerProduct_forward_inplace          Layer_forward_inplace

#endif // LAYER_INNERPRODUCT_H
