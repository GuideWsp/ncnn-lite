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

#include "dropout.h"
#include <math.h>

void *Dropout_ctor(void *_self, va_list *args)
{
    Layer *self = (Layer *)_self;

    self->one_blob_only = true;
    self->support_inplace = true;

    return _self;
}

int Dropout_load_param(void *_self, const ParamDict& pd)
{
    Dropout *self = (Dropout *)_self;

    self->scale = pd.get(0, 1.f);

    return 0;
}

int Dropout_forward_inplace(void *_self, Mat& bottom_top_blob, const Option& opt)
{
    Dropout *self = (Dropout *)_self;

    if (self->scale == 1.f)
    {
        return 0;
    }

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i=0; i<size; i++)
        {
            ptr[i] = ptr[i] * self->scale;
        }
    }

    return 0;
}
