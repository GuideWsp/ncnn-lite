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

#include "innerproduct.h"
#include <algorithm>
#include "layer_type.h"

#include "cstl/utils.h"

void *InnerProduct_ctor(void *_self, va_list *args)
{
    Layer *self = (Layer *)_self;

    self->one_blob_only = true;
    self->support_inplace = false;

    return _self;
}

int InnerProduct_load_param(void *_self, const ParamDict& pd)
{
    InnerProduct *self = (InnerProduct *)_self;

    self->num_output = pd.get(0, 0);
    self->bias_term = pd.get(1, 0);
    self->weight_data_size = pd.get(2, 0);
    self->int8_scale_term = pd.get(8, 0);
    self->activation_type = pd.get(9, 0);
    self->activation_params = pd.get(10, Mat());

    return 0;
}

int InnerProduct_load_model(void *_self, const ModelBin& mb)
{
    InnerProduct *self = (InnerProduct *)_self;

    self->weight_data = mb.load(self->weight_data_size, 0);
    if (self->weight_data.empty())
        return -100;

    if (self->bias_term)
    {
        self->bias_data = mb.load(self->num_output, 1);
        if (self->bias_data.empty())
            return -100;
    }

    if (self->int8_scale_term)
    {
        self->weight_data_int8_scales = mb.load(self->num_output, 1);
        self->bottom_blob_int8_scale = mb.load(1, 1)[0];
    }

    return 0;
}

int InnerProduct_create_pipeline(void *_self, const Option& opt)
{
    InnerProduct *self = (InnerProduct *)_self;

    // runtime quantize the weight data
    if (opt.use_int8_inference && self->weight_data.elemsize == (size_t)4u && self->int8_scale_term)
    {
        Mat int8_weight_data(self->weight_data_size, (size_t)1u);
        if (int8_weight_data.empty())
            return -100;

        const int weight_data_size_output = self->weight_data_size / self->num_output;

        for (int p=0; p<self->num_output; p++)
        {
            Option opt_q = opt;
            opt_q.blob_allocator = int8_weight_data.allocator;

            const Mat weight_data_n = self->weight_data.range(weight_data_size_output * p, weight_data_size_output);
            Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * p, weight_data_size_output);
            quantize_float32_to_int8(weight_data_n, int8_weight_data_n, self->weight_data_int8_scales[p], opt_q);
        }

        self->weight_data = int8_weight_data;
    }

    return 0;
}

int InnerProduct_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    InnerProduct *self = (InnerProduct *)_self;

    if (opt.use_int8_inference && self->weight_data.elemsize == (size_t)1u)
    {
        return InnerProduct_forward_int8(self, bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    top_blob.create(self->num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<self->num_output; p++)
    {
        float sum = 0.f;

        if (self->bias_term)
            sum = self->bias_data[p];

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* w = (const float*)self->weight_data + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        if (self->activation_type == 1)
        {
            sum = max(sum, 0.f);
        }
        else if (self->activation_type == 2)
        {
            float slope = self->activation_params[0];
            sum = sum > 0.f ? sum : sum * slope;
        }
        else if (self->activation_type == 3)
        {
            float min = self->activation_params[0];
            float max = self->activation_params[1];
            if (sum < min)
                sum = min;
            if (sum > max)
                sum = max;
        }
        else if (self->activation_type == 4)
        {
            sum = static_cast<float>(1.f / (1.f + exp(-sum)));
        }

        top_blob[p] = sum;
    }

    return 0;
}

int InnerProduct_forward_int8(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    InnerProduct *self = (InnerProduct *)_self;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    Mat bottom_blob_tm = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_float32_to_int8(bottom_blob, bottom_blob_tm, self->bottom_blob_int8_scale, opt_g);
    }

    top_blob.create(self->num_output, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<self->num_output; p++)
    {
        float* outptr = top_blob;

        int sum = 0;

        // channels
        for (int q=0; q<channels; q++)
        {
            const signed char* w = (const signed char*)self->weight_data + size * channels * p + size * q;
            const signed char* m = bottom_blob_tm.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        // dequantize and relu
        float scale_in;
        if (self->weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (self->bottom_blob_int8_scale * self->weight_data_int8_scales[p]);

        float sumfp32 = sum * scale_in;

        if (self->bias_term)
            sumfp32 += self->bias_data[p];

        if (self->activation_type == 1)
        {
            sumfp32 = max(sumfp32, 0.f);
        }

        outptr[p] = sumfp32;
    }

    return 0;
}
