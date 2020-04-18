// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (self->c) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "reshape.h"

void *Reshape_ctor(void *_self, va_list *args)
{
    Layer *layer = (Layer *)_self;

    layer->one_blob_only = true;
    layer->support_inplace = false;

    return _self;
}

int Reshape_load_param(void *_self, const ParamDict& pd)
{
    Reshape *self = (Reshape *)_self;

    self->w = pd.get(0, -233);
    self->h = pd.get(1, -233);
    self->c = pd.get(2, -233);
    self->permute = pd.get(3, 0);

    self->ndim = 3;
    if (self->c == -233)
        self->ndim = 2;
    if (self->h == -233)
        self->ndim = 1;
    if (self->w == -233)
        self->ndim = 0;

    return 0;
}

int Reshape_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Reshape *self = (Reshape *)_self;

    size_t elemsize = bottom_blob.elemsize;
    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c;

    if (self->ndim == 1)
    {
        int _w = self->w;

        if (_w == 0)
            _w = bottom_blob.w;

        if (_w == -1)
            _w = total;

        if (self->permute == 1)
        {
            top_blob.create(_w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            // self->c-self->h-self->w to self->h-self->w-self->c
            float* ptr = top_blob;
            for (int i=0; i<bottom_blob.h; i++)
            {
                for (int j=0; j<bottom_blob.w; j++)
                {
                    for (int p=0; p<bottom_blob.c; p++)
                    {
                        const float* bptr = bottom_blob.channel(p);
                        *ptr++ = bptr[i*bottom_blob.w + j];
                    }
                }
            }
        }
        else
        {
            top_blob = bottom_blob.reshape(_w, opt.blob_allocator);
        }
    }
    else if (self->ndim == 2)
    {
        int _w = self->w;
        int _h = self->h;

        if (_w == 0)
            _w = bottom_blob.w;
        if (_h == 0)
            _h = bottom_blob.h;

        if (_w == -1)
            _w = total / _h;
        if (_h == -1)
            _h = total / _w;

        top_blob = bottom_blob.reshape(_w, _h, opt.blob_allocator);
    }
    else if (self->ndim == 3)
    {
        int _w = self->w;
        int _h = self->h;
        int _c = self->c;

        if (_w == 0)
            _w = bottom_blob.w;
        if (_h == 0)
            _h = bottom_blob.h;
        if (_c == 0)
            _c = bottom_blob.c;

        if (_w == -1)
            _w = total / _c / _h;
        if (_h == -1)
            _h = total / _c / _w;
        if (_c == -1)
            _c = total / _h / _w;

        top_blob = bottom_blob.reshape(_w, _h, _c, opt.blob_allocator);
    }

    if (top_blob.empty())
        return -100;

    return 0;
}
