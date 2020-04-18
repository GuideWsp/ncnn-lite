// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "packing.h"

void *Packing_ctor(void *_self, va_list *args)
{
    Layer *self = (Layer *)_self;

    self->one_blob_only = true;
    self->support_inplace = false;

    return _self;
}

int Packing_load_param(void *_self, const ParamDict& pd)
{
    Packing *self = (Packing *)_self;

    self->out_elempack = pd.get(0, 1);
    self->use_padding = pd.get(1, 0);

    return 0;
}

int Packing_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Packing *self = (Packing *)_self;

    int elempack = bottom_blob.elempack;

    if (elempack == self->out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    if (!self->use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % self->out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % self->out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 3 && channels * elempack % self->out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        if (self->out_elempack == 1)
        {
            top_blob = bottom_blob;
            top_blob.w = w * elempack;
            top_blob.cstep = w * elempack;
            top_blob.elemsize = elemsize / elempack;
            top_blob.elempack = self->out_elempack;
            return 0;
        }

        int outw = (w * elempack + self->out_elempack - 1) / self->out_elempack;
        size_t out_elemsize = elemsize / elempack * self->out_elempack;

        top_blob.create(outw, out_elemsize, self->out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        memcpy(top_blob.data, bottom_blob.data, w * elemsize);

        return 0;
    }

    if (dims == 2)
    {
        int outh = (h * elempack + self->out_elempack - 1) / self->out_elempack;
        size_t out_elemsize = elemsize / elempack * self->out_elempack;
        size_t lane_size = out_elemsize / self->out_elempack;

        top_blob.create(w, outh, out_elemsize, self->out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int i = 0; i < outh; i++)
        {
            unsigned char* outptr = (unsigned char*)top_blob + i * w * out_elemsize;

            for (int j = 0; j < w; j++)
            {
                unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                for (int k = 0; k < self->out_elempack; k++)
                {
                    int srcy = (i * self->out_elempack + k) / elempack;
                    if (srcy >= h)
                        break;

                    int srck = (i * self->out_elempack + k) % elempack;

                    const unsigned char* ptr = (const unsigned char*)bottom_blob + srcy * w * elemsize;
                    const unsigned char* elem_ptr = ptr + j * elemsize;

                    memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        int outc = (channels * elempack + self->out_elempack - 1) / self->out_elempack;
        size_t out_elemsize = elemsize / elempack * self->out_elempack;
        size_t lane_size = out_elemsize / self->out_elempack;

        top_blob.create(w, h, outc, out_elemsize, self->out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int q = 0; q < outc; q++)
        {
            Mat out = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                unsigned char* outptr = (unsigned char*)out + i * w * out_elemsize;

                for (int j = 0; j < w; j++)
                {
                    unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                    for (int k = 0; k < self->out_elempack; k++)
                    {
                        int srcq = (q * self->out_elempack + k) / elempack;
                        if (srcq >= channels)
                            break;

                        int srck = (q * self->out_elempack + k) % elempack;

                        const Mat m = bottom_blob.channel(srcq);
                        const unsigned char* ptr = (const unsigned char*)m + i * w * elemsize;
                        const unsigned char* elem_ptr = ptr + j * elemsize;

                        memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                    }
                }
            }
        }

        return 0;
    }

    return 0;
}
