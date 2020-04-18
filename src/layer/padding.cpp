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

#include "padding.h"

void *Padding_ctor(void *_self, va_list *args)
{
    Layer *self = (Layer *)_self;

    self->one_blob_only = true;
    self->support_inplace = false;

    return _self;
}

int Padding_load_param(void *_self, const ParamDict& pd)
{
    Padding *self = (Padding *)_self;
    Layer *layer = (Layer *)_self;

    self->top = pd.get(0, 0);
    self->bottom = pd.get(1, 0);
    self->left = pd.get(2, 0);
    self->right = pd.get(3, 0);
    self->type = pd.get(4, 0);
    self->value = pd.get(5, 0.f);
    self->per_channel_pad_data_size = pd.get(6, 0);

    if (self->top == -233 && self->bottom == -233 && self->left == -233 && self->right == -233)
    {
        layer->one_blob_only = false;
    }

    return 0;
}

int Padding_load_model(void *_self, const ModelBin& mb)
{
    Padding *self = (Padding *)_self;

    if (self->per_channel_pad_data_size)
    {
        self->per_channel_pad_data = mb.load(self->per_channel_pad_data_size, 1);
    }

    return 0;
}

template<typename T>
static void copy_make_border_image(const Mat& src, Mat& dst, int top, int left, int type, T v)
{
    int w = dst.w;
    int h = dst.h;

    const T* ptr = src;
    T* outptr = dst;

    if (type == 0)
    {
        int y = 0;
        // fill top
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            outptr += w;
        }
        // fill center
        for (; y < (top + src.h); y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = v;
            }
            if (src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            ptr += src.w;
            outptr += w;
        }
        // fill bottom
        for (; y < h; y++)
        {
            int x = 0;
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            outptr += w;
        }
    }

    if (type == 1)
    {
        int y = 0;
        // fill top
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            if(src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - 1];
            }
            outptr += w;
        }
        // fill center
        for (; y < (top + src.h); y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            if(src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - 1];
            }
            ptr += src.w;
            outptr += w;
        }
        // fill bottom
        ptr -= src.w;
        for (; y < h; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            if(src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - 1];
            }
            outptr += w;
        }
    }

    if (type == 2)
    {
        int y = 0;
        // fill top
        ptr += top * src.w;
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }
            if(src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - (x - left - src.w) - 2];
            }
            outptr += w;
            ptr -= src.w;
        }
        // fill center
        for (; y < (top + src.h); y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }
            if(src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - (x - left - src.w) - 2];
            }
            ptr += src.w;
            outptr += w;
        }
        // fill bottom
        ptr -= 2 * src.w;
        for (; y < h; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }
            if(src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - (x - left - src.w) - 2];
            }
            outptr += w;
            ptr -= src.w;
        }
    }

}

int Padding_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Padding *self = (Padding *)_self;

    if (self->top == 0 && self->bottom == 0 && self->left == 0 && self->right == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w + self->left + self->right;

    if (dims == 1)
    {
        top_blob.create(outw, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_make_border_image<signed char>(bottom_blob, top_blob, 0, self->left, self->type, static_cast<signed char>(self->value));
        if (elemsize == 2)
            copy_make_border_image<unsigned short>(bottom_blob, top_blob, 0, self->left, self->type, float32_to_bfloat16(self->value));
        if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, 0, self->left, self->type, self->value);

        return 0;
    }

    int outh = h + self->top + self->bottom;

    if (dims == 2)
    {
        top_blob.create(outw, outh, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_make_border_image<signed char>(bottom_blob, top_blob, self->top, self->left, self->type, static_cast<signed char>(self->value));
        if (elemsize == 2)
            copy_make_border_image<unsigned short>(bottom_blob, top_blob, self->top, self->left, self->type, float32_to_bfloat16(self->value));
        if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, self->top, self->left, self->type, self->value);

        return 0;
    }

    if (dims == 3)
    {
        top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob.channel(q);
            Mat borderm = top_blob.channel(q);

            float pad_value = self->per_channel_pad_data_size ? self->per_channel_pad_data[q] : self->value;

            if (elemsize == 1)
                copy_make_border_image<signed char>(m, borderm, self->top, self->left, self->type, static_cast<signed char>(pad_value));
            if (elemsize == 2)
                copy_make_border_image<unsigned short>(m, borderm, self->top, self->left, self->type, float32_to_bfloat16(pad_value));
            if (elemsize == 4)
                copy_make_border_image<float>(m, borderm, self->top, self->left, self->type, pad_value);
        }

        return 0;
    }

    return 0;
}

int Padding_forward_multi(void *_self, const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt)
{
    Padding *self = (Padding *)_self;

    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];

    int _top;
    int _bottom;
    int _left;
    int _right;
    {
        const int* param_data = reference_blob;

        _top = param_data[0];
        _bottom = param_data[1];
        _left = param_data[2];
        _right = param_data[3];
    }

    if (_top == 0 && _bottom == 0 && _left == 0 && _right == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w + _left + _right;

    if (dims == 1)
    {
        top_blob.create(outw, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_make_border_image<signed char>(bottom_blob, top_blob, 0, _left, self->type, static_cast<signed char>(self->value));
        if (elemsize == 2)
            copy_make_border_image<unsigned short>(bottom_blob, top_blob, 0, _left, self->type, float32_to_bfloat16(self->value));
        if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, 0, _left, self->type, self->value);

        return 0;
    }

    int outh = h + _top + _bottom;

    if (dims == 2)
    {
        top_blob.create(outw, outh, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_make_border_image<signed char>(bottom_blob, top_blob, _top, _left, self->type, static_cast<signed char>(self->value));
        if (elemsize == 2)
            copy_make_border_image<unsigned short>(bottom_blob, top_blob, _top, _left, self->type, float32_to_bfloat16(self->value));
        if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, _top, _left, self->type, self->value);

        return 0;
    }

    if (dims == 3)
    {
        top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob.channel(q);
            Mat borderm = top_blob.channel(q);

            float pad_value = self->per_channel_pad_data_size ? self->per_channel_pad_data[q] : self->value;

            if (elemsize == 1)
                copy_make_border_image<signed char>(m, borderm, _top, _left, self->type, static_cast<signed char>(pad_value));
            if (elemsize == 2)
                copy_make_border_image<unsigned short>(m, borderm, _top, _left, self->type, float32_to_bfloat16(pad_value));
            if (elemsize == 4)
                copy_make_border_image<float>(m, borderm, _top, _left, self->type, pad_value);
        }

        return 0;
    }

    return 0;
}
