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

#ifndef LAYER_PADDING_H
#define LAYER_PADDING_H

#include "layer.h"

struct Padding
{
    // layer base
    Layer layer;

    // proprietary data
    // -233 = dynamic offset from reference blob
    int top;
    int bottom;
    int left;
    int right;
    int type;// 0=CONSTANT 1=REPLICATE 2=REFLECT
    float value;

    // per channel pad value
    int per_channel_pad_data_size;
    Mat per_channel_pad_data;
};

void *Padding_ctor(void *_self, va_list *args);

int Padding_load_param(void *_self, const ParamDict& pd);

int Padding_load_model(void *_self, const ModelBin& mb);

int Padding_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

int Padding_forward_multi(void *_self, const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt);

// default operators
#define Padding_dtor                     Layer_dtor
#define Padding_create_pipeline          Layer_create_pipeline
#define Padding_destroy_pipeline         Layer_destroy_pipeline
#define Padding_forward_inplace_multi    Layer_forward_inplace_multi
#define Padding_forward_inplace          Layer_forward_inplace

#endif // LAYER_PADDING_H
