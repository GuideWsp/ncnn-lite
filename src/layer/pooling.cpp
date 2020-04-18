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

#include "pooling.h"
#include <float.h>
#include <algorithm>
#include "layer_type.h"

#include "cstl/utils.h"

void *Pooling_ctor(void *_self, va_list *args)
{
    Layer *self = (Layer *)_self;

    self->one_blob_only = true;
    self->support_inplace = false;

    return _self;
}

int Pooling_load_param(void *_self, const ParamDict& pd)
{
    Pooling *self = (Pooling *)_self;

    self->pooling_type = pd.get(0, 0);
    self->kernel_w = pd.get(1, 0);
    self->kernel_h = pd.get(11, self->kernel_w);
    self->stride_w = pd.get(2, 1);
    self->stride_h = pd.get(12, self->stride_w);
    self->pad_left = pd.get(3, 0);
    self->pad_right = pd.get(14, self->pad_left);
    self->pad_top = pd.get(13, self->pad_left);
    self->pad_bottom = pd.get(15, self->pad_top);
    self->global_pooling = pd.get(4, 0);
    self->pad_mode = pd.get(5, 0);
    self->avgpool_count_include_pad = pd.get(6, 0);

    return 0;
}

int Pooling_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Pooling *self = (Pooling *)_self;

    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

//     fprintf(stderr, "Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);
    if (self->global_pooling)
    {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (self->pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float max = ptr[0];
                for (int i=0; i<size; i++)
                {
                    max = max(max, ptr[i]);
                }

                top_blob[q] = max;
            }
        }
        else if (self->pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                top_blob[q] = sum / size;
            }
        }

        return 0;
    }

    Mat bottom_blob_bordered;
    Pooling_make_padding(self, bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - self->kernel_w) / self->stride_w + 1;
    int outh = (h - self->kernel_h) / self->stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int maxk = self->kernel_w * self->kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - self->kernel_w;
        for (int i = 0; i < self->kernel_h; i++)
        {
            for (int j = 0; j < self->kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (self->pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*self->stride_h) + j*self->stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        max = max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (self->pooling_type == PoolMethod_AVE)
    {
        if (self->avgpool_count_include_pad == 0)
        {
            int wtailpad = 0;
            int htailpad = 0;

            if (self->pad_mode == 0) // full padding
            {
                wtailpad = bottom_blob_bordered.w - bottom_blob.w - self->pad_left - self->pad_right;
                htailpad = bottom_blob_bordered.h - bottom_blob.h - self->pad_top - self->pad_bottom;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * self->stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * self->stride_w;

                        float sum = 0;
                        int area = 0;

                        for (int ki = 0; ki < self->kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < self->pad_top)
                                continue;

                            if (sy >= h - self->pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < self->kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < self->pad_left)
                                    continue;

                                if (sx >= w - self->pad_right - wtailpad)
                                    break;

                                float val = m.row(sy)[sx];
                                sum += val;
                                area += 1;
                            }
                        }

                        outptr[j] = sum / area;
                    }

                    outptr += outw;
                }
            }
        }
        else // if (avgpool_count_include_pad == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i*self->stride_h) + j*self->stride_w;

                        float sum = 0;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = sptr[ space_ofs[k] ];
                            sum += val;
                        }

                        outptr[j] = sum / maxk;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

void Pooling_make_padding(void *_self, const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt)
{
    Pooling *self = (Pooling *)_self;

    int w = bottom_blob.w;
    int h = bottom_blob.h;

    bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (self->pooling_type == PoolMethod_MAX)
    {
        pad_value = -FLT_MAX;
    }
    else if (self->pooling_type == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    int wtailpad = 0;
    int htailpad = 0;

    if (self->pad_mode == 0) // full padding
    {
        int wtail = (w + self->pad_left + self->pad_right - self->kernel_w) % self->stride_w;
        int htail = (h + self->pad_top + self->pad_bottom - self->kernel_h) % self->stride_h;

        if (wtail != 0)
            wtailpad = self->stride_w - wtail;
        if (htail != 0)
            htailpad = self->stride_h - htail;

        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, self->pad_top, self->pad_bottom + htailpad, self->pad_left, self->pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (self->pad_mode == 1) // valid padding
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, self->pad_top, self->pad_bottom, self->pad_left, self->pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (self->pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
    {
        int wpad = self->kernel_w + (w - 1) / self->stride_w * self->stride_w - w;
        int hpad = self->kernel_h + (h - 1) / self->stride_h * self->stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (self->pad_mode == 3) // onnx padding=SAME_LOWER
    {
        int wpad = self->kernel_w + (w - 1) / self->stride_w * self->stride_w - w;
        int hpad = self->kernel_h + (h - 1) / self->stride_h * self->stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}
