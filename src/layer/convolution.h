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

#ifndef LAYER_CONVOLUTION_H
#define LAYER_CONVOLUTION_H

#include "layer.h"

struct Convolution
{
    // layer base
    Layer layer;

    // proprietary data
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left;// -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    int pad_top;
    int pad_bottom;
    float pad_value;
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
    float top_blob_int8_scale;// TODO load param

    bool use_int8_requantize;

    // implementation type, 0 means do not use auto pack model 
    int impl_type;
};

void *Convolution_ctor(void *_self, va_list *args);

int Convolution_load_param(void *_self, const ParamDict& pd);

int Convolution_load_model(void *_self, const ModelBin& mb);

int Convolution_create_pipeline(void *_self, const Option& opt);

int Convolution_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

void Convolution_make_padding(void *_self, const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt);

int Convolution_forward_int8(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

// default operators
#define Convolution_dtor                     Layer_dtor
#define Convolution_destroy_pipeline         Layer_destroy_pipeline
#define Convolution_forward_multi            Layer_forward_multi
#define Convolution_forward_inplace_multi    Layer_forward_inplace_multi
#define Convolution_forward_inplace          Layer_forward_inplace

#endif // LAYER_CONVOLUTION_H
