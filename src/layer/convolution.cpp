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

#include "convolution.h"
#include <algorithm>
#include "layer_type.h"

#include "cstl/utils.h"

void *Convolution_ctor(void *_self, va_list *args)
{
    Convolution *self = (Convolution *)_self;

    self->layer.one_blob_only = true;
    self->layer.support_inplace = false;

    self->use_int8_requantize = false;

    return _self;
}

int Convolution_load_param(void *_self, const ParamDict& pd)
{
    Convolution *self = (Convolution *)_self;

    self->num_output = pd.get(0, 0);
    self->kernel_w = pd.get(1, 0);
    self->kernel_h = pd.get(11, self->kernel_w);
    self->dilation_w = pd.get(2, 1);
    self->dilation_h = pd.get(12, self->dilation_w);
    self->stride_w = pd.get(3, 1);
    self->stride_h = pd.get(13, self->stride_w);
    self->pad_left = pd.get(4, 0);
    self->pad_right = pd.get(15, self->pad_left);
    self->pad_top = pd.get(14, self->pad_left);
    self->pad_bottom = pd.get(16, self->pad_top);
    self->pad_value = pd.get(18, 0.f);
    self->bias_term = pd.get(5, 0);
    self->weight_data_size = pd.get(6, 0);
    self->int8_scale_term = pd.get(8, 0);
    self->activation_type = pd.get(9, 0);
    self->activation_params = pd.get(10, Mat());
    self->impl_type = pd.get(17, 0);

    return 0;
}

int Convolution_load_model(void *_self, const ModelBin& mb)
{
    Convolution *self = (Convolution *)_self;

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

int Convolution_create_pipeline(void *_self, const Option& opt)
{
    Convolution *self = (Convolution *)_self;

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

int Convolution_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Convolution *self = (Convolution *)_self;

    // convolv with NxN kernel
    // value = value + bias

    if (opt.use_int8_inference && self->weight_data.elemsize == (size_t)1u)
    {
        return Convolution_forward_int8(self, bottom_blob, top_blob, opt);
    }

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && self->kernel_w == 1 && self->kernel_h == 1)
    {
        int num_input = self->weight_data_size / self->num_output;
        if (bottom_blob.w == num_input)
        {
            // call InnerProduct
            Layer* op = create_layer(LayerInnerProduct);

            // set param
            ParamDict pd;
            pd.set(0, self->num_output);
            pd.set(1, self->bias_term);
            pd.set(2, self->weight_data_size);
            pd.set(8, self->int8_scale_term);

            op->load_param(op, pd);

            // set weights
            Mat weights[4];
            weights[0] = self->weight_data;
            weights[1] = self->bias_data;

            if (self->int8_scale_term)
            {
                weights[2] = self->weight_data_int8_scales;
                weights[3] = Mat(1, (size_t)4u, (void*)&self->bottom_blob_int8_scale);
            }

            op->load_model(op, ModelBinFromMatArray(weights));

            op->create_pipeline(op, opt);

            // forward
            op->forward(op, bottom_blob, top_blob, opt);

            cdelete(op);

            return 0;
        }
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

//     fprintf(stderr, "Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = self->dilation_w * (self->kernel_w - 1) + 1;
    const int kernel_extent_h = self->dilation_h * (self->kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    Convolution_make_padding(self, bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / self->stride_w + 1;
    int outh = (h - kernel_extent_h) / self->stride_h + 1;

    const int maxk = self->kernel_w * self->kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * self->dilation_h - self->kernel_w * self->dilation_w;
        for (int i = 0; i < self->kernel_h; i++)
        {
            for (int j = 0; j < self->kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += self->dilation_w;
            }
            p2 += gap;
        }
    }

    // float32
    top_blob.create(outw, outh, self->num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<self->num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (self->bias_term)
                    sum = self->bias_data[p];

                const float* kptr = (const float*)self->weight_data + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const float* sptr = m.row(i*self->stride_h) + j*self->stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[ space_ofs[k] ]; // 20.72
                        float w = kptr[k];
                        sum += val * w; // 41.45
                    }

                    kptr += maxk;
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

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }

    return 0;
}

void Convolution_make_padding(void *_self, const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt)
{
    Convolution *self = (Convolution *)_self;

    int w = bottom_blob.w;
    int h = bottom_blob.h;

    const int kernel_extent_w = self->dilation_w * (self->kernel_w - 1) + 1;
    const int kernel_extent_h = self->dilation_h * (self->kernel_h - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (self->pad_left > 0 || self->pad_right > 0 || self->pad_top > 0 || self->pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, self->pad_top, self->pad_bottom, self->pad_left, self->pad_right, BORDER_CONSTANT, self->pad_value, opt_b);
    }
    else if (self->pad_left == -233 && self->pad_right == -233 && self->pad_top == -233 && self->pad_bottom == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / self->stride_w * self->stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / self->stride_h * self->stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, self->pad_value, opt_b);
        }
    }
    else if (self->pad_left == -234 && self->pad_right == -234 && self->pad_top == -234 && self->pad_bottom == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / self->stride_w * self->stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / self->stride_h * self->stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, self->pad_value, opt_b);
        }
    }
}

static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

int Convolution_forward_int8(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Convolution *self = (Convolution *)_self;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

//     fprintf(stderr, "Convolution input %d x %d  ksize=%d %d  stride=%d %d\n", w, h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = self->dilation_w * (self->kernel_w - 1) + 1;
    const int kernel_extent_h = self->dilation_h * (self->kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_float32_to_int8(bottom_blob, bottom_blob_unbordered, self->bottom_blob_int8_scale, opt_g);
    }

    Mat bottom_blob_bordered;
    Convolution_make_padding(self, bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / self->stride_w + 1;
    int outh = (h - kernel_extent_h) / self->stride_h + 1;

    const int maxk = self->kernel_w * self->kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * self->dilation_h - self->kernel_w * self->dilation_w;
        for (int i = 0; i < self->kernel_h; i++)
        {
            for (int j = 0; j < self->kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += self->dilation_w;
            }
            p2 += gap;
        }
    }

    // int8
    size_t out_elemsize = self->use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, self->num_output, out_elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<self->num_output; p++)
    {
        signed char* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                int sum = 0;

                const signed char* kptr = (const signed char*)self->weight_data + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const signed char* sptr = m.row<signed char>(i * self->stride_h) + j * self->stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        int val = sptr[ space_ofs[k] ];
                        int w = kptr[k];
                        sum += val * w;
                    }

                    kptr += maxk;
                }

                if (self->use_int8_requantize)
                {
                    // requantize and relu
                    float scale_in;
                    if (self->weight_data_int8_scales[p] == 0)
                        scale_in = 0;
                    else
                        scale_in = 1.f / (self->bottom_blob_int8_scale * self->weight_data_int8_scales[p]);

                    float sumfp32 = sum * scale_in;

                    if (self->bias_term)
                        sumfp32 += self->bias_data[p];

                    float scale_out = self->top_blob_int8_scale;//FIXME load param

                    signed char sums8 = float2int8(sumfp32 * scale_out);

                    if (self->activation_type == 1)
                    {
                        sums8 = max(sums8, (signed char)0);
                    }

                    outptr[0] = sums8;
                    outptr += 1;
                }
                else
                {
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

                    ((float*)outptr)[0] = sumfp32;
                    outptr += 4;
                }
            }
        }
    }

    return 0;
}
