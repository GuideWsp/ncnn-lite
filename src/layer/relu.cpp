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

#include "relu.h"
#include <algorithm>

void *ReLU_ctor(void *_self, va_list *args)
{
    Layer *self = (Layer *)_self;

    self->one_blob_only = true;
    self->support_inplace = true;

    return _self;
}

int ReLU_load_param(void *_self, const ParamDict& pd)
{
    ReLU *self = (ReLU *)_self;

    self->slope = pd.get(0, 0.f);

    return 0;
}

int ReLU_forward_inplace(void *_self, Mat& bottom_top_blob, const Option& opt)
{
    ReLU *self = (ReLU *)_self;

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (self->slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            signed char* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }
    else
    {
        // TODO
        // #pragma omp parallel for num_threads(opt.num_threads)
        // for (int q=0; q<channels; q++)
        // {
        //     float* ptr = bottom_top_blob.channel(q);

        //     for (int i=0; i<size; i++)
        //     {
        //         if (ptr[i] < 0)
        //             ptr[i] *= slope;
        //     }
        // }
    }

    return 0;
}

int ReLU_forward_inplace_int8(void *_self, Mat& bottom_top_blob, const Option& opt)
{
    ReLU *self = (ReLU *)_self;

    if (bottom_top_blob.elemsize == 1u)
        return ReLU_forward_inplace_int8(self, bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (self->slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                {
                    ptr[i] *= self->slope;
                }
            }
        }
    }

    return 0;
}
