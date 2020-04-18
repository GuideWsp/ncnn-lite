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

#include "convolution_arm.h"
#include "benchmark.h"
#include "cpu.h"

#include "layer_type.h"

#include "cstl/utils.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#include "neon_activation.h"
#endif // __ARM_NEON

#include "convolution_1x1.h"
#include "convolution_2x2.h"
#include "convolution_3x3.h"
#include "convolution_4x4.h"
#include "convolution_5x5.h"
#include "convolution_7x7.h"
#include "convolution_sgemm.h"
#include "convolution_sgemm_int8.h"
#include "convolution_1x1_int8.h"
#include "convolution_3x3_int8.h"

#include "convolution_1x1_bf16s.h"

#if __ARM_NEON
#include "convolution_1x1_pack4.h"
#include "convolution_1x1_pack4to1.h"
#include "convolution_3x3_pack4.h"
#include "convolution_3x3_pack1to4.h"
#include "convolution_3x3_pack4to1.h"
#include "convolution_5x5_pack4.h"
#include "convolution_7x7_pack1to4.h"

#include "convolution_1x1_pack4_bf16s.h"
#include "convolution_1x1_pack4to1_bf16s.h"
#include "convolution_3x3_pack4_bf16s.h"
#include "convolution_3x3_pack1to4_bf16s.h"
#include "convolution_3x3_pack4to1_bf16s.h"
#include "convolution_5x5_pack4_bf16s.h"
#include "convolution_7x7_pack1to4_bf16s.h"
#endif // __ARM_NEON

void *Convolution_arm_ctor(void *_self, va_list *args)
{
    Convolution_arm *self = (Convolution_arm *)_self;
    Layer *layer = (Layer *)_self;

#if __ARM_NEON
    layer->support_packing = true;
#endif // __ARM_NEON

    layer->support_bf16_storage = true;

    self->activation = 0;
    self->convolution_dilation1 = 0;

    return _self;
}

int Convolution_arm_create_pipeline(void *_self, const Option& opt)
{
    Convolution_arm *self = (Convolution_arm *)_self;
    Convolution *parent = (Convolution *)_self;
    Layer *layer = (Layer *)_self;

    if (parent->activation_type == 1)
    {
        self->activation = create_layer(LayerReLU);

        ParamDict pd;
        self->activation->load_param(self->activation, pd);
    }
    else if (parent->activation_type == 2)
    {
        self->activation = create_layer(LayerReLU);

        ParamDict pd;
        pd.set(0, parent->activation_params[0]);// slope
        self->activation->load_param(self->activation, pd);
    }
    else if (parent->activation_type == 3)
    {
        self->activation = create_layer(LayerClip);

        ParamDict pd;
        pd.set(0, parent->activation_params[0]);// min
        pd.set(1, parent->activation_params[1]);// max
        self->activation->load_param(self->activation, pd);
    }
    else if (parent->activation_type == 4)
    {
        self->activation = create_layer(LayerSigmoid);

        ParamDict pd;
        self->activation->load_param(self->activation, pd);
    }

    if (self->activation)
    {
        self->activation->create_pipeline(self->activation, opt);
    }

    if (opt.use_bf16_storage)
    {
        return Convolution_arm_create_pipeline_bf16s(self, opt);
    }

    if (opt.use_int8_inference && parent->weight_data.elemsize == (size_t)1u)
    {
        layer->support_packing = false;

        return Convolution_arm_create_pipeline_int8_arm(self, opt);
    }

    if (opt.use_packing_layout == false && parent->kernel_w == parent->kernel_h && parent->dilation_w != 1 && parent->dilation_h == parent->dilation_w && parent->stride_w == 1 && parent->stride_h == 1)
    {
        self->convolution_dilation1 = create_layer(LayerConvolution);

        // set param
        ParamDict pd;
        pd.set(0, parent->num_output);// parent->num_output
        pd.set(1, parent->kernel_w);
        pd.set(11, parent->kernel_h);
        pd.set(2, 1);
        pd.set(12, 1);
        pd.set(3, 1);// parent->stride_w
        pd.set(13, 1);// parent->stride_h
        pd.set(4, 0);// pad_w
        pd.set(14, 0);// pad_h
        pd.set(5, parent->bias_term);
        pd.set(6, parent->weight_data_size);

        self->convolution_dilation1->load_param(self->convolution_dilation1, pd);

        // set weights
        if (parent->bias_term)
        {
            Mat weights[2];
            weights[0] = parent->weight_data;
            weights[1] = parent->bias_data;

            self->convolution_dilation1->load_model(self->convolution_dilation1, ModelBinFromMatArray(weights));
        }
        else
        {
            Mat weights[1];
            weights[0] = parent->weight_data;

            self->convolution_dilation1->load_model(self->convolution_dilation1, ModelBinFromMatArray(weights));
        }

        self->convolution_dilation1->create_pipeline(self->convolution_dilation1, opt);

        return 0;
    }

    const int maxk = parent->kernel_w * parent->kernel_h;
    const int num_input = parent->weight_data_size / maxk / parent->num_output;

    int elempack = (opt.use_packing_layout && num_input % 4 == 0) ? 4 : 1;
    int out_elempack = (opt.use_packing_layout && parent->num_output % 4 == 0) ? 4 : 1;

#if __ARM_NEON
    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_neon(parent->weight_data, weight_data_pack4, num_input, parent->num_output);
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_neon(parent->weight_data, weight_data_pack4, num_input, parent->num_output);
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4_neon(parent->weight_data, weight_data_pack4, num_input, parent->num_output);
        }
        else
        {
            // src = kw-kh-inch-outch
            // dst = 4b-4a-kw-kh-inch/4a-outch/4b
            Mat weight_data_r2 = parent->weight_data.reshape(maxk, num_input, parent->num_output);

            weight_data_pack4.create(maxk, num_input/4, parent->num_output/4, (size_t)4*16, 16);

            for (int q=0; q+3<parent->num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack4.channel(q/4);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    const float* k10 = k1.row(p);
                    const float* k11 = k1.row(p+1);
                    const float* k12 = k1.row(p+2);
                    const float* k13 = k1.row(p+3);

                    const float* k20 = k2.row(p);
                    const float* k21 = k2.row(p+1);
                    const float* k22 = k2.row(p+2);
                    const float* k23 = k2.row(p+3);

                    const float* k30 = k3.row(p);
                    const float* k31 = k3.row(p+1);
                    const float* k32 = k3.row(p+2);
                    const float* k33 = k3.row(p+3);

                    float* g00 = g0.row(p/4);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k10[k];
                        g00[2] = k20[k];
                        g00[3] = k30[k];

                        g00[4] = k01[k];
                        g00[5] = k11[k];
                        g00[6] = k21[k];
                        g00[7] = k31[k];

                        g00[8] = k02[k];
                        g00[9] = k12[k];
                        g00[10] = k22[k];
                        g00[11] = k32[k];

                        g00[12] = k03[k];
                        g00[13] = k13[k];
                        g00[14] = k23[k];
                        g00[15] = k33[k];

                        g00 += 16;
                    }
                }
            }
        }
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-kw-kh-inch-outch/4b
        {
            Mat weight_data_r2 = parent->weight_data.reshape(maxk, num_input, parent->num_output);

            weight_data_pack1to4.create(maxk, num_input, parent->num_output/4, (size_t)4*4, 4);

            for (int q=0; q+3<parent->num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack1to4.channel(q/4);

                for (int p=0; p<num_input; p++)
                {
                    const float* k00 = k0.row(p);
                    const float* k10 = k1.row(p);
                    const float* k20 = k2.row(p);
                    const float* k30 = k3.row(p);

                    float* g00 = g0.row(p);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k10[k];
                        g00[2] = k20[k];
                        g00[3] = k30[k];

                        g00 += 4;
                    }
                }
            }
        }
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_neon(parent->weight_data, weight_data_pack4to1, num_input, parent->num_output);
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_neon(parent->weight_data, weight_data_pack4to1, num_input, parent->num_output);
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4to1_neon(parent->weight_data, weight_data_pack4to1, num_input, parent->num_output);
        }
        else
        {
            // src = kw-kh-inch-outch
            // dst = 4a-kw-kh-inch/4a-outch
            Mat weight_data_r2 = parent->weight_data.reshape(maxk, num_input, parent->num_output);

            weight_data_pack4to1.create(maxk, num_input/4, parent->num_output, (size_t)4*4, 4);

            for (int q=0; q<parent->num_output; q++)
            {
                const Mat k0 = weight_data_r2.channel(q);
                Mat g0 = weight_data_pack4to1.channel(q);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    float* g00 = g0.row(p/4);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k01[k];
                        g00[2] = k02[k];
                        g00[3] = k03[k];

                        g00 += 4;
                    }
                }
            }
        }
    }
#endif // __ARM_NEON

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        self->use_winograd3x3 = false;
        self->use_sgemm1x1 = false;

        if (opt.use_winograd_convolution && parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            // winograd is slow on small channel count
            if (num_input >= 16 && parent->num_output >= 16)
                self->use_winograd3x3 = true;

            if (self->use_winograd3x3)
            {
//                 conv3x3s1_winograd64_transform_kernel_neon(parent->weight_data, self->weight_3x3_winograd64_data, num_input, parent->num_output);
                conv3x3s1_winograd64_transform_kernel_neon5(parent->weight_data, self->weight_3x3_winograd64_data, num_input, parent->num_output);
            }
        }

        // TODO assume more proper condition
        if (opt.use_sgemm_convolution && parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            if (num_input >= 64 && parent->num_output >= 64)
                self->use_sgemm1x1 = true;

            if (self->use_sgemm1x1)
            {
                conv1x1s1_sgemm_transform_kernel_neon(parent->weight_data, self->weight_1x1_sgemm_data, num_input, parent->num_output);
            }
        }

        if (parent->impl_type > 0 && parent->impl_type < 6 && parent->impl_type != 4)
        {
            switch (parent->impl_type)
            {
                case 1:
                    // winograd
                    conv3x3s1_winograd64_transform_kernel_neon5(parent->weight_data, self->weight_3x3_winograd64_data, num_input, parent->num_output);
                    break;
                case 2:
                    // pointwise
                    conv1x1s1_sgemm_transform_kernel_neon(parent->weight_data, self->weight_1x1_sgemm_data, num_input, parent->num_output);
                    break;
                case 3:
                    // im2col
                    conv_im2col_sgemm_transform_kernel_neon(parent->weight_data, self->weight_sgemm_data, num_input, parent->num_output, maxk);
                    break;
//                 case 4:
//                     // direct
//                     break;
                case 5:
                    // conv3x3s2
                    conv3x3s2_transform_kernel_neon(parent->weight_data, self->weight_3x3s2_data, num_input, parent->num_output);
                    break;
            }
        }

        if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv3x3s2_transform_kernel_neon(parent->weight_data, self->weight_3x3s2_data, num_input, parent->num_output);
        }

        if (opt.use_sgemm_convolution && parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv_im2col_sgemm_transform_kernel_neon(parent->weight_data, self->weight_sgemm_data, num_input, parent->num_output, maxk);
        }

        if (opt.use_sgemm_convolution && parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv_im2col_sgemm_transform_kernel_neon(parent->weight_data, self->weight_sgemm_data, num_input, parent->num_output, maxk);
        }
    }

    return 0;
}

int Convolution_arm_destroy_pipeline(void *_self, const Option& opt)
{
    Convolution_arm *self = (Convolution_arm *)_self;
    Convolution *parent = (Convolution *)_self;

    if (self->activation)
    {
        self->activation->destroy_pipeline(self->activation, opt);
        cdelete(self->activation);
        self->activation = 0;
    }

    if (self->convolution_dilation1)
    {
        self->convolution_dilation1->destroy_pipeline(self->convolution_dilation1, opt);
        cdelete(self->convolution_dilation1);
        self->convolution_dilation1 = 0;
    }

    return 0;
}

int Convolution_arm_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Convolution_arm *self = (Convolution_arm *)_self;
    Convolution *parent = (Convolution *)_self;

    if (bottom_blob.dims != 3)
    {
        return Convolution_forward(self, bottom_blob, top_blob, opt);
    }

    if (opt.use_int8_inference && parent->weight_data.elemsize == (size_t)1u)
    {
        return Convolution_arm_forward_int8_arm(self, bottom_blob, top_blob, opt);
    }

    if (opt.use_bf16_storage)
        return Convolution_arm_forward_bf16s(self, bottom_blob, top_blob, opt);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

//     fprintf(stderr, "Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, parent->kernel_w, parent->kernel_h, parent->stride_w, parent->stride_h);

    const int kernel_extent_w = parent->dilation_w * (parent->kernel_w - 1) + 1;
    const int kernel_extent_h = parent->dilation_h * (parent->kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    Convolution_make_padding(self, bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / parent->stride_w + 1;
    int outh = (h - kernel_extent_h) / parent->stride_h + 1;
    int out_elempack = (opt.use_packing_layout && parent->num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, parent->num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (opt.use_packing_layout == false && parent->kernel_w == parent->kernel_h && parent->dilation_w != 1 && parent->dilation_h == parent->dilation_w && parent->stride_w == 1 && parent->stride_h == 1)
    {
        return Convolution_arm_forwardDilation_arm(self, bottom_blob_bordered, top_blob, opt);
    }

    const int maxk = parent->kernel_w * parent->kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * parent->dilation_h - parent->kernel_w * parent->dilation_w;
        for (int i = 0; i < parent->kernel_h; i++)
        {
            for (int j = 0; j < parent->kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += parent->dilation_w;
            }
            p2 += gap;
        }
    }

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv1x1s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv3x3s1_winograd64_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv3x3s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 5 && parent->kernel_h == 5 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv5x5s1_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 5 && parent->kernel_h == 5 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv5x5s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // parent->num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<parent->num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)parent->bias_data) + p * 4);
                        }

                        const float* kptr = (const float*)weight_data_pack4 + maxk * channels * p * 16;

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i*parent->stride_h) + j*parent->stride_w * 4;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                float32x4_t _val = vld1q_f32( sptr + space_ofs[k] * 4 );

                                float32x4_t _w0 = vld1q_f32( kptr );
                                float32x4_t _w1 = vld1q_f32( kptr + 4 );
                                float32x4_t _w2 = vld1q_f32( kptr + 8 );
                                float32x4_t _w3 = vld1q_f32( kptr + 12 );

#if __aarch64__
                                _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                                _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                                _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                                _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
#else
                                _sum = vmlaq_lane_f32(_sum, _w0, vget_low_f32(_val), 0);
                                _sum = vmlaq_lane_f32(_sum, _w1, vget_low_f32(_val), 1);
                                _sum = vmlaq_lane_f32(_sum, _w2, vget_high_f32(_val), 0);
                                _sum = vmlaq_lane_f32(_sum, _w3, vget_high_f32(_val), 1);
#endif

                                kptr += 16;
                            }
                        }

                        _sum = activation_ps(_sum, parent->activation_type, parent->activation_params);

                        vst1q_f32(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv3x3s1_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv3x3s2_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 7 && parent->kernel_h == 7 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv7x7s2_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // parent->num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<parent->num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)parent->bias_data) + p * 4);
                        }

                        const float* kptr = (const float*)weight_data_pack1to4 + maxk * channels * p * 4;

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i*parent->stride_h) + j*parent->stride_w;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                float32x4_t _val = vdupq_n_f32( sptr[ space_ofs[k] ] );
                                float32x4_t _w = vld1q_f32( kptr );
                                _sum = vmlaq_f32(_sum, _val, _w);

                                kptr += 4;
                            }
                        }

                        _sum = activation_ps(_sum, parent->activation_type, parent->activation_params);

                        vst1q_f32(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv1x1s2_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd64_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, parent->bias_data, opt);

//             conv3x3s1_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // parent->num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<parent->num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = parent->bias_data[p];
                        }

                        const float* kptr = (const float*)weight_data_pack4to1 + maxk * channels * p * 4;

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i*parent->stride_h) + j*parent->stride_w * 4;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                float32x4_t _val = vld1q_f32( sptr + space_ofs[k] * 4 );
                                float32x4_t _w = vld1q_f32( kptr );
                                float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                                sum += vaddvq_f32(_s4); // dot
#else
                                float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                                _ss = vpadd_f32(_ss, _ss);
                                sum += vget_lane_f32(_ss, 0);
#endif

                                kptr += 4;
                            }
                        }

                        sum = activation_ss(sum, parent->activation_type, parent->activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        if (parent->impl_type > 0 && parent->impl_type < 6 && parent->impl_type != 4)
        {
            // engineering is magic.
            switch (parent->impl_type)
            {
                case 1:
                    conv3x3s1_winograd64_neon5(bottom_blob_bordered, top_blob, self->weight_3x3_winograd64_data, parent->bias_data, opt);
                    break;
                case 2:
                    conv1x1s1_sgemm_neon(bottom_blob_bordered, top_blob, self->weight_1x1_sgemm_data, parent->bias_data, opt);
                    break;
                case 3:
                    conv_im2col_sgemm_neon(bottom_blob_bordered, top_blob, self->weight_sgemm_data, parent->bias_data, parent->kernel_w, parent->kernel_h, parent->stride_w, parent->stride_h, opt);
                    break;
//                 case 4: FIXME fallback to auto path
//                     conv(bottom_blob_bordered, top_blob, parent->weight_data, parent->bias_data, opt);
//                     break;
                case 5:
                    conv3x3s2_packed_neon(bottom_blob_bordered, top_blob, self->weight_3x3s2_data, parent->bias_data, opt);
                    break;
            }

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            if (self->use_sgemm1x1)
            {
                conv1x1s1_sgemm_neon(bottom_blob_bordered, top_blob, self->weight_1x1_sgemm_data, parent->bias_data, opt);
            }
            else
            {
                conv1x1s1_neon(bottom_blob_bordered, top_blob, parent->weight_data, parent->bias_data, opt);
            }

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            if (opt.use_sgemm_convolution)
                conv_im2col_sgemm_neon(bottom_blob_bordered, top_blob, self->weight_sgemm_data, parent->bias_data, parent->kernel_w, parent->kernel_h, parent->stride_w, parent->stride_h, opt);
            else
                conv1x1s2_neon(bottom_blob_bordered, top_blob, parent->weight_data, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            if (self->use_winograd3x3 && w <= 120 && h <= 120)
            {
//                 conv3x3s1_winograd64_neon4(bottom_blob_bordered, top_blob, self->weight_3x3_winograd64_data, parent->bias_data, opt);
                conv3x3s1_winograd64_neon5(bottom_blob_bordered, top_blob, self->weight_3x3_winograd64_data, parent->bias_data, opt);
            }
            else
            {
                conv3x3s1_neon(bottom_blob_bordered, top_blob, parent->weight_data, parent->bias_data, opt);
            }

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            if (opt.use_sgemm_convolution && !(outw >=8 && outh >=8))
                conv_im2col_sgemm_neon(bottom_blob_bordered, top_blob, self->weight_sgemm_data, parent->bias_data, parent->kernel_w, parent->kernel_h, parent->stride_w, parent->stride_h, opt);
            else
                conv3x3s2_packed_neon(bottom_blob_bordered, top_blob, self->weight_3x3s2_data, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else if (parent->kernel_w == 4 && parent->kernel_h == 4 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 4 && parent->stride_h == 4)
        {
            conv4x4s4_neon(bottom_blob_bordered, top_blob, parent->weight_data, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else if (parent->kernel_w == 5 && parent->kernel_h == 5 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv5x5s1_neon(bottom_blob_bordered, top_blob, parent->weight_data, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else if (parent->kernel_w == 5 && parent->kernel_h == 5 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv5x5s2_neon(bottom_blob_bordered, top_blob, parent->weight_data, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else if (parent->kernel_w == 7 && parent->kernel_h == 7 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv7x7s1_neon(bottom_blob_bordered, top_blob, parent->weight_data, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else if (parent->kernel_w == 7 && parent->kernel_h == 7 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv7x7s2_neon(bottom_blob_bordered, top_blob, parent->weight_data, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else
        {
            // parent->num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<parent->num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (parent->bias_term)
                        {
                            sum = parent->bias_data[p];
                        }

                        const float* kptr = (const float*)parent->weight_data + maxk * channels * p;

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i*parent->stride_h) + j*parent->stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[ space_ofs[k] ];
                                float w = kptr[ k ];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        if (parent->activation_type == 1)
                        {
                            sum = max(sum, 0.f);
                        }
                        else if (parent->activation_type == 2)
                        {
                            float slope = parent->activation_params[0];
                            sum = sum > 0.f ? sum : sum * slope;
                        }
                        else if (parent->activation_type == 3)
                        {
                            float min = parent->activation_params[0];
                            float max = parent->activation_params[1];
                            if (sum < min)
                                sum = min;
                            if (sum > max)
                                sum = max;
                        }
                        else if (parent->activation_type == 4)
                        {
                            sum = static_cast<float>(1.f / (1.f + exp(-sum)));
                        }

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

int Convolution_arm_create_pipeline_bf16s(void *_self, const Option& opt)
{
    Convolution_arm *self = (Convolution_arm *)_self;
    Convolution *parent = (Convolution *)_self;

    const int maxk = parent->kernel_w * parent->kernel_h;
    const int num_input = parent->weight_data_size / maxk / parent->num_output;

    int elempack = (opt.use_packing_layout && num_input % 4 == 0) ? 4 : 1;
    int out_elempack = (opt.use_packing_layout && parent->num_output % 4 == 0) ? 4 : 1;

#if __ARM_NEON
    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_bf16s_neon(parent->weight_data, weight_data_pack4_bf16, num_input, parent->num_output);
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_bf16s_neon(parent->weight_data, weight_data_pack4_bf16, num_input, parent->num_output);
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4_neon(parent->weight_data, weight_data_pack4_bf16, num_input, parent->num_output);
        }
        else
        {
            // src = kw-kh-inch-outch
            // dst = 4b-4a-kw-kh-inch/4a-outch/4b
            Mat weight_data_r2 = parent->weight_data.reshape(maxk, num_input, parent->num_output);

            weight_data_pack4_bf16.create(maxk, num_input/4, parent->num_output/4, (size_t)2*16, 16);

            for (int q=0; q+3<parent->num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack4_bf16.channel(q/4);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    const float* k10 = k1.row(p);
                    const float* k11 = k1.row(p+1);
                    const float* k12 = k1.row(p+2);
                    const float* k13 = k1.row(p+3);

                    const float* k20 = k2.row(p);
                    const float* k21 = k2.row(p+1);
                    const float* k22 = k2.row(p+2);
                    const float* k23 = k2.row(p+3);

                    const float* k30 = k3.row(p);
                    const float* k31 = k3.row(p+1);
                    const float* k32 = k3.row(p+2);
                    const float* k33 = k3.row(p+3);

                    unsigned short* g00 = g0.row<unsigned short>(p/4);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00[1] = float32_to_bfloat16(k10[k]);
                        g00[2] = float32_to_bfloat16(k20[k]);
                        g00[3] = float32_to_bfloat16(k30[k]);

                        g00[4] = float32_to_bfloat16(k01[k]);
                        g00[5] = float32_to_bfloat16(k11[k]);
                        g00[6] = float32_to_bfloat16(k21[k]);
                        g00[7] = float32_to_bfloat16(k31[k]);

                        g00[8] = float32_to_bfloat16(k02[k]);
                        g00[9] = float32_to_bfloat16(k12[k]);
                        g00[10] = float32_to_bfloat16(k22[k]);
                        g00[11] = float32_to_bfloat16(k32[k]);

                        g00[12] = float32_to_bfloat16(k03[k]);
                        g00[13] = float32_to_bfloat16(k13[k]);
                        g00[14] = float32_to_bfloat16(k23[k]);
                        g00[15] = float32_to_bfloat16(k33[k]);

                        g00 += 16;
                    }
                }
            }
        }
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-kw-kh-inch-outch/4b
        {
            Mat weight_data_r2 = parent->weight_data.reshape(maxk, num_input, parent->num_output);

            weight_data_pack1to4_bf16.create(maxk, num_input, parent->num_output/4, (size_t)2*4, 4);

            for (int q=0; q+3<parent->num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack1to4_bf16.channel(q/4);

                for (int p=0; p<num_input; p++)
                {
                    const float* k00 = k0.row(p);
                    const float* k10 = k1.row(p);
                    const float* k20 = k2.row(p);
                    const float* k30 = k3.row(p);

                    unsigned short* g00 = g0.row<unsigned short>(p);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00[1] = float32_to_bfloat16(k10[k]);
                        g00[2] = float32_to_bfloat16(k20[k]);
                        g00[3] = float32_to_bfloat16(k30[k]);

                        g00 += 4;
                    }
                }
            }
        }
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_bf16s_neon(parent->weight_data, weight_data_pack4to1_bf16, num_input, parent->num_output);
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_bf16s_neon(parent->weight_data, weight_data_pack4to1_bf16, num_input, parent->num_output);
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4to1_neon(parent->weight_data, weight_data_pack4to1_bf16, num_input, parent->num_output);
        }
        else
        {
            // src = kw-kh-inch-outch
            // dst = 4a-kw-kh-inch/4a-outch
            Mat weight_data_r2 = parent->weight_data.reshape(maxk, num_input, parent->num_output);

            weight_data_pack4to1_bf16.create(maxk, num_input/4, parent->num_output, (size_t)2*4, 4);

            for (int q=0; q<parent->num_output; q++)
            {
                const Mat k0 = weight_data_r2.channel(q);
                Mat g0 = weight_data_pack4to1_bf16.channel(q);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    unsigned short* g00 = g0.row<unsigned short>(p/4);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00[1] = float32_to_bfloat16(k01[k]);
                        g00[2] = float32_to_bfloat16(k02[k]);
                        g00[3] = float32_to_bfloat16(k03[k]);

                        g00 += 4;
                    }
                }
            }
        }
    }
#endif // __ARM_NEON

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_bf16s_neon(parent->weight_data, self->weight_data_bf16, num_input, parent->num_output);
        }
        else
        {
            cast_float32_to_bfloat16(parent->weight_data, self->weight_data_bf16, opt);
        }
    }

    return 0;
}

int Convolution_arm_forward_bf16s(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Convolution_arm *self = (Convolution_arm *)_self;
    Convolution *parent = (Convolution *)_self;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

//     fprintf(stderr, "Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, parent->kernel_w, parent->kernel_h, parent->stride_w, parent->stride_h);

    const int kernel_extent_w = parent->dilation_w * (parent->kernel_w - 1) + 1;
    const int kernel_extent_h = parent->dilation_h * (parent->kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    Convolution_make_padding(self, bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / parent->stride_w + 1;
    int outh = (h - kernel_extent_h) / parent->stride_h + 1;
    int out_elempack = (opt.use_packing_layout && parent->num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, parent->num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // FIXME
//     if (opt.use_packing_layout == false && parent->kernel_w == parent->kernel_h && parent->dilation_w != 1 && parent->dilation_h == parent->dilation_w && parent->stride_w == 1 && parent->stride_h == 1)
//     {
//         return forwardDilation_arm(bottom_blob_bordered, top_blob, opt);
//     }

    const int maxk = parent->kernel_w * parent->kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * parent->dilation_h - parent->kernel_w * parent->dilation_w;
        for (int i = 0; i < parent->kernel_h; i++)
        {
            for (int j = 0; j < parent->kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += parent->dilation_w;
            }
            p2 += gap;
        }
    }

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv1x1s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv3x3s1_winograd64_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv3x3s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 5 && parent->kernel_h == 5 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv5x5s1_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 5 && parent->kernel_h == 5 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv5x5s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // parent->num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<parent->num_output / out_elempack; p++)
            {
                unsigned short* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)parent->bias_data) + p * 4);
                        }

                        const unsigned short* kptr = weight_data_pack4_bf16.channel(p);

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const unsigned short* sptr = m.row<const unsigned short>(i*parent->stride_h) + j*parent->stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( sptr + space_ofs[k] * 4 ), 16));

                                float32x4_t _w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( kptr ), 16));
                                float32x4_t _w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( kptr + 4 ), 16));
                                float32x4_t _w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( kptr + 8 ), 16));
                                float32x4_t _w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( kptr + 12 ), 16));

#if __aarch64__
                                _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                                _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                                _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                                _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
#else
                                _sum = vmlaq_lane_f32(_sum, _w0, vget_low_f32(_val), 0);
                                _sum = vmlaq_lane_f32(_sum, _w1, vget_low_f32(_val), 1);
                                _sum = vmlaq_lane_f32(_sum, _w2, vget_high_f32(_val), 0);
                                _sum = vmlaq_lane_f32(_sum, _w3, vget_high_f32(_val), 1);
#endif

                                kptr += 16;
                            }
                        }

                        _sum = activation_ps(_sum, parent->activation_type, parent->activation_params);

                        vst1_u16(outptr + j * 4, vshrn_n_u32(vreinterpretq_u32_f32(_sum), 16));
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv3x3s1_pack1to4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv3x3s2_pack1to4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 7 && parent->kernel_h == 7 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv7x7s2_pack1to4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // parent->num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<parent->num_output / out_elempack; p++)
            {
                unsigned short* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)parent->bias_data) + p * 4);
                        }

                        const unsigned short* kptr = weight_data_pack1to4_bf16.channel(p);

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const unsigned short* sptr = m.row<const unsigned short>(i*parent->stride_h) + j*parent->stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vdupq_n_f32(bfloat16_to_float32( sptr[ space_ofs[k] ] ));
                                float32x4_t _w = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( kptr ), 16));
                                _sum = vmlaq_f32(_sum, _val, _w);

                                kptr += 4;
                            }
                        }

                        _sum = activation_ps(_sum, parent->activation_type, parent->activation_params);

                        vst1_u16(outptr + j * 4, vshrn_n_u32(vreinterpretq_u32_f32(_sum), 16));
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_pack4to1_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv1x1s2_pack4to1_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd64_pack4to1_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1_bf16, parent->bias_data, opt);

//             conv3x3s1_pack4to1_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // parent->num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<parent->num_output; p++)
            {
                unsigned short* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = parent->bias_data[p];
                        }

                        const unsigned short* kptr = weight_data_pack4to1_bf16.channel(p);

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const unsigned short* sptr = m.row<const unsigned short>(i*parent->stride_h) + j*parent->stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( sptr + space_ofs[k] * 4 ), 16));
                                float32x4_t _w = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( kptr ), 16));
                                float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                                sum += vaddvq_f32(_s4); // dot
#else
                                float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                                _ss = vpadd_f32(_ss, _ss);
                                sum += vget_lane_f32(_ss, 0);
#endif

                                kptr += 4;
                            }
                        }

                        sum = activation_ss(sum, parent->activation_type, parent->activation_params);

                        outptr[j] = float32_to_bfloat16(sum);
                    }

                    outptr += outw;
                }
            }
        }
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
        {
            conv1x1s1_sgemm_bf16s_neon(bottom_blob_bordered, top_blob, self->weight_data_bf16, parent->bias_data, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }
        }
        else
        {
            // parent->num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<parent->num_output; p++)
            {
                unsigned short* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (parent->bias_term)
                        {
                            sum = parent->bias_data[p];
                        }

                        const unsigned short* kptr = (const unsigned short*)self->weight_data_bf16 + maxk * channels * p;

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const unsigned short* sptr = m.row<unsigned short>(i*parent->stride_h) + j*parent->stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = bfloat16_to_float32(sptr[ space_ofs[k] ]);
                                float w = bfloat16_to_float32(kptr[ k ]);
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        if (parent->activation_type == 1)
                        {
                            sum = max(sum, 0.f);
                        }
                        else if (parent->activation_type == 2)
                        {
                            float slope = parent->activation_params[0];
                            sum = sum > 0.f ? sum : sum * slope;
                        }
                        else if (parent->activation_type == 3)
                        {
                            float min = parent->activation_params[0];
                            float max = parent->activation_params[1];
                            if (sum < min)
                                sum = min;
                            if (sum > max)
                                sum = max;
                        }
                        else if (parent->activation_type == 4)
                        {
                            sum = static_cast<float>(1.f / (1.f + exp(-sum)));
                        }

                        outptr[j] = float32_to_bfloat16(sum);
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

int Convolution_arm_create_pipeline_int8_arm(void *_self, const Option& opt)
{
    Convolution_arm *self = (Convolution_arm *)_self;
    Convolution *parent = (Convolution *)_self;

    const int maxk = parent->kernel_w * parent->kernel_h;
    const int num_input = parent->weight_data_size / maxk / parent->num_output;

    self->use_winograd3x3_int8 = false;
    self->use_sgemm1x1_int8 = false;

    if (opt.use_winograd_convolution && parent->kernel_w == 3 && parent->kernel_h == 3 &&
        parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
    {
        self->use_winograd3x3_int8 = true;
//         conv3x3s1_winograd23_transform_kernel_int8_neon(parent->weight_data, self->weight_3x3_winograd23_data_int8, num_input, parent->num_output);
        conv3x3s1_winograd43_transform_kernel_int8_neon(parent->weight_data, self->weight_3x3_winograd23_data_int8, num_input, parent->num_output);
    }

    if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
    {
        conv3x3s2_transform_kernel_int8_neon(parent->weight_data, self->weight_3x3s2_data_int8, num_input, parent->num_output);
    }
    else if (parent->kernel_w == 1 && parent->kernel_h == 1 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 1 && parent->stride_h == 1)
    {
        self->use_sgemm1x1_int8 = true;
        conv1x1s1_sgemm_transform_kernel_int8_neon(parent->weight_data, self->weight_1x1s1_sgemm_data_int8, num_input, parent->num_output);
    }
    else
    {
        conv_im2col_sgemm_transform_kernel_int8_neon(parent->weight_data, self->weight_sgemm_data_int8, num_input, parent->num_output, maxk);
    }

    return 0;
}

int Convolution_arm_forward_int8_arm(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Convolution_arm *self = (Convolution_arm *)_self;
    Convolution *parent = (Convolution *)_self;

    if (parent->dilation_w > 1 || parent->dilation_h > 1)
    {
        return Convolution_forward(self, bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    // int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

//     fprintf(stderr, "Convolution_arm input %d x %d  ksize=%d %d  stride=%d %d\n", w, h, parent->kernel_w, parent->kernel_h, parent->stride_w, parent->stride_h);

    const int kernel_extent_w = parent->dilation_w * (parent->kernel_w - 1) + 1;
    const int kernel_extent_h = parent->dilation_h * (parent->kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_float32_to_int8(bottom_blob, bottom_blob_unbordered, parent->bottom_blob_int8_scale, opt_g);
    }

    Mat bottom_blob_bordered;
    Convolution_make_padding(self, bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / parent->stride_w + 1;
    int outh = (h - kernel_extent_h) / parent->stride_h + 1;

    // int8
    size_t out_elemsize = parent->use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, parent->num_output, out_elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // int8
    if (parent->use_int8_requantize == true)
    {
        Mat top_blob_tm;
        top_blob_tm.create(outw, outh, parent->num_output, (size_t)4u, opt.workspace_allocator);
        if (top_blob_tm.empty())
            return -100;
        
        if (self->use_sgemm1x1_int8)
        {
            std::vector<float> requantize_scales;
            for (int p=0; p<parent->num_output; p++)
            {
                float scale_in;
                if (parent->weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (parent->bottom_blob_int8_scale * parent->weight_data_int8_scales[p]);

                float scale_out = parent->top_blob_int8_scale;

                requantize_scales.push_back(scale_in);
                requantize_scales.push_back(scale_out);
            }

            conv1x1s1_sgemm_int8_requant_neon(bottom_blob_bordered, top_blob, self->weight_1x1s1_sgemm_data_int8, parent->bias_data, requantize_scales, opt);

            if (self->activation)
            {
                self->activation->forward_inplace(self->activation, top_blob, opt);
            }

            return 0;
        }
        else if (self->use_winograd3x3_int8)
        {
//             conv3x3s1_winograd23_int8_neon(bottom_blob_bordered, top_blob_tm, self->weight_3x3_winograd23_data_int8, opt);
            conv3x3s1_winograd43_int8_neon(bottom_blob_bordered, top_blob_tm, self->weight_3x3_winograd23_data_int8, opt);
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 &&
                 parent->dilation_w == 1 && parent->dilation_h == 1 &&
                 parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv3x3s2_packed_int8_neon(bottom_blob_bordered, top_blob_tm, self->weight_3x3s2_data_int8, opt);
        }
        else
        {
            conv_im2col_sgemm_int8_neon(bottom_blob_bordered, top_blob_tm, self->weight_sgemm_data_int8, parent->kernel_w, parent->kernel_h, parent->stride_w, parent->stride_h, opt);
        }

        // requantize, reverse scale inplace
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<parent->num_output; p++)
        {
            Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = top_blob.allocator;

            Mat top_blob_tm_g = top_blob_tm.channel_range(p, 1);
            Mat top_blob_g = top_blob.channel_range(p, 1);

            // requantize and relu
            float scale_in;
            if (parent->weight_data_int8_scales[p] == 0)
                scale_in = 0;
            else
                scale_in = 1.f / (parent->bottom_blob_int8_scale * parent->weight_data_int8_scales[p]);

            float scale_out = parent->top_blob_int8_scale;//FIXME load param

            requantize_int8_to_int8(top_blob_tm_g, top_blob_g, scale_in, scale_out, parent->bias_term ? (const float*)parent->bias_data + p : 0, parent->bias_term ? 1 : 0, 0, opt_g);
        }
    }
    else
    {
        if (self->use_sgemm1x1_int8)
        {
            conv1x1s1_sgemm_int8_neon(bottom_blob_bordered, top_blob, self->weight_1x1s1_sgemm_data_int8, opt);
        }
        else if (self->use_winograd3x3_int8)
        {
//             conv3x3s1_winograd23_int8_neon(bottom_blob_bordered, top_blob, self->weight_3x3_winograd23_data_int8, opt);
            conv3x3s1_winograd43_int8_neon(bottom_blob_bordered, top_blob, self->weight_3x3_winograd23_data_int8, opt);
        }
        else if (parent->kernel_w == 3 && parent->kernel_h == 3 && parent->dilation_w == 1 && parent->dilation_h == 1 && parent->stride_w == 2 && parent->stride_h == 2)
        {
            conv3x3s2_packed_int8_neon(bottom_blob_bordered, top_blob, self->weight_3x3s2_data_int8, opt);
        }
        else
        {
            conv_im2col_sgemm_int8_neon(bottom_blob_bordered, top_blob, self->weight_sgemm_data_int8, parent->kernel_w, parent->kernel_h, parent->stride_w, parent->stride_h, opt);
        }

        // dequantize, reverse scale inplace
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<parent->num_output; p++)
        {
            Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = top_blob.allocator;

            Mat top_blob_g = top_blob.channel_range(p, 1);

            // dequantize
            float scale_in;
            if (parent->weight_data_int8_scales[p] == 0)
                scale_in = 0;
            else
                scale_in = 1.f / (parent->bottom_blob_int8_scale * parent->weight_data_int8_scales[p]);

            dequantize_int32_to_float32(top_blob_g, scale_in, parent->bias_term ? (const float*)parent->bias_data + p : 0, parent->bias_term ? 1 : 0, opt_g);
        }
    }

    if (self->activation)
    {
        self->activation->forward_inplace(self->activation, top_blob, opt);
    }           

    return 0;
}

int Convolution_arm_forwardDilation_arm(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Convolution_arm *self = (Convolution_arm *)_self;
    Convolution *parent = (Convolution *)_self;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_size = parent->kernel_w;
    const int stride = parent->stride_w;
    const int dilation = parent->dilation_w;
    const int kernel_extent = dilation * (kernel_size - 1) + 1;

    int outw = (w - kernel_extent) / stride + 1;
    int outh = (h - kernel_extent) / stride + 1;

    top_blob.create(outw, outh, parent->num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Make (dilation * dilation) batches
    Mat inner_bottom_blob;
    Mat inner_top_blob;
    for (int x = 0; x < dilation; x ++)
    {
        for (int y = 0; y < dilation; y ++)
        {
            int inner_w = (w - y + dilation - 1) / dilation;
            int inner_h = (h - x + dilation - 1) / dilation;

            int inner_outw = (inner_w - kernel_size) / stride + 1;
            int inner_outh = (inner_h - kernel_size) / stride + 1;

            inner_bottom_blob.create(inner_w, inner_h, bottom_blob.c, elemsize, opt.workspace_allocator);
            if (inner_bottom_blob.empty())
                return -100;

            inner_top_blob.create(inner_outw, inner_outh, parent->num_output, elemsize, opt.workspace_allocator);
            if (inner_top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < bottom_blob.c; c ++)
            {
                float *outptr = inner_bottom_blob.channel(c);

                for (int i = 0; i < inner_h; i ++)
                {
                    const float *ptr = (const float *) bottom_blob.channel(c) + dilation * i * w + x * w + y;
                    for (int j = 0; j < inner_w; j ++)
                    {
                        outptr[j] = ptr[j*dilation];
                    }
                    outptr += inner_w;
                }
            }

            Option opt_g = opt;
            opt_g.blob_allocator = inner_top_blob.allocator;
            self->convolution_dilation1->forward(self->convolution_dilation1, inner_bottom_blob, inner_top_blob, opt_g);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < parent->num_output; c ++)
            {
                float *outptr = (float *) top_blob.channel(c) + x * outw + y;
                for (int i = 0; i < inner_outh; i ++)
                {
                    const float *ptr = (const float *) inner_top_blob.channel(c) + i * inner_outw;
                    for (int j = 0; j < inner_outw; j ++)
                    {
                        outptr[j*dilation] = ptr[j];
                    }
                    outptr += dilation * outw;
                }
            }
        }
    }

    if (self->activation)
    {
        self->activation->forward_inplace(self->activation, top_blob, opt);
    }

    return 0;
}
