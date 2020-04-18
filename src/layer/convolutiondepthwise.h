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

#ifndef LAYER_CONVOLUTIONDEPTHWISE_H
#define LAYER_CONVOLUTIONDEPTHWISE_H

#include "layer.h"

struct ConvolutionDepthWise
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
    int group;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    // model
    Mat weight_data;
    Mat bias_data;

    Mat weight_data_int8_scales;
    Mat bottom_blob_int8_scales;
    float top_blob_int8_scale;

    bool use_int8_requantize;
};

void *ConvolutionDepthWise_ctor(void *_self, va_list *args);

int ConvolutionDepthWise_load_param(void *_self, const ParamDict& pd);

int ConvolutionDepthWise_load_model(void *_self, const ModelBin& mb);

int ConvolutionDepthWise_create_pipeline(void *_self, const Option& opt);

int ConvolutionDepthWise_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

void ConvolutionDepthWise_make_padding(void *_self, const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt);

int ConvolutionDepthWise_forward_int8(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

// default operators
#define ConvolutionDepthWise_dtor                     Layer_dtor
#define ConvolutionDepthWise_destroy_pipeline         Layer_destroy_pipeline
#define ConvolutionDepthWise_forward_multi            Layer_forward_multi
#define ConvolutionDepthWise_forward_inplace_multi    Layer_forward_inplace_multi
#define ConvolutionDepthWise_forward_inplace          Layer_forward_inplace

#endif // LAYER_CONVOLUTIONDEPTHWISE_H
