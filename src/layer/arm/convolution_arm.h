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

#ifndef LAYER_CONVOLUTION_ARM_H
#define LAYER_CONVOLUTION_ARM_H

#include "convolution.h"

struct Convolution_arm
{
    // layer base
    Convolution layer;

    // proprietary data
    Layer* activation;
    bool use_winograd3x3;
    bool use_sgemm1x1;
    Mat weight_3x3_winograd64_data;
    Mat weight_1x1_sgemm_data;
    Mat weight_3x3s2_data;
    Mat weight_sgemm_data;

    // forwardDilation
    Layer* convolution_dilation1;

    // pack4
    Mat weight_data_pack4;
    Mat weight_data_pack1to4;
    Mat weight_data_pack4to1;

    // bf16
    Mat weight_data_pack4_bf16;
    Mat weight_data_pack1to4_bf16;
    Mat weight_data_pack4to1_bf16;
    Mat weight_data_bf16;

    // int8
    bool use_winograd3x3_int8;
    bool use_sgemm1x1_int8;
    Mat weight_3x3s2_data_int8;
    Mat weight_1x1s1_sgemm_data_int8;
    Mat weight_sgemm_data_int8;
    std::vector<Mat> weight_3x3_winograd23_data_int8;
};

void *Convolution_arm_ctor(void *_self, va_list *args);

int Convolution_arm_create_pipeline(void *_self, const Option& opt);

int Convolution_arm_destroy_pipeline(void *_self, const Option& opt);

int Convolution_arm_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

int Convolution_arm_create_pipeline_bf16s(void *_self, const Option& opt);
int Convolution_arm_forward_bf16s(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);
int Convolution_arm_create_pipeline_int8_arm(void *_self, const Option& opt);
int Convolution_arm_forward_int8_arm(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);
int Convolution_arm_forwardDilation_arm(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

// default operators
#define Convolution_arm_dtor                     Layer_dtor
#define Convolution_arm_load_param               Layer_load_param
#define Convolution_arm_load_model               Layer_load_model
#define Convolution_arm_forward_multi            Layer_forward_multi
#define Convolution_arm_forward_inplace_multi    Layer_forward_inplace_multi
#define Convolution_arm_forward_inplace          Layer_forward_inplace

#endif // LAYER_CONVOLUTION_ARM_H
