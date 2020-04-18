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

#include "layer.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include "cpu.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif
#include "layer_declaration.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

void *Layer_ctor(void *_self, va_list *args)
{
    Layer *self = (Layer *)_self;

    self->load_param = va_arg(*args, int (*)(void*, const ParamDict&));
    self->load_model = va_arg(*args, int (*)(void*, const ModelBin&));
    self->create_pipeline = va_arg(*args, int (*)(void*, const Option&));
    self->destroy_pipeline = va_arg(*args, int (*)(void*, const Option&));
    self->forward_multi = va_arg(*args, int (*)(void*, const std::vector<Mat>&, std::vector<Mat>&, const Option&));
    self->forward = va_arg(*args, int (*)(void*, const Mat&, Mat&, const Option&));
    self->forward_inplace_multi = va_arg(*args, int (*)(void*, std::vector<Mat>&, const Option&));
    self->forward_inplace = va_arg(*args, int (*)(void*, Mat&, const Option&));

    self->one_blob_only = false;
    self->support_inplace = false;
    self->support_packing = false;

    self->support_bf16_storage = false;

    return _self;
}

// layer destructor
void *Layer_dtor(void *_self)
{
    return _self;
}

int Layer_create_pipeline(void *_self, const Option& opt)
{
    return 0;
}

int Layer_destroy_pipeline(void *_self, const Option& opt)
{
    return 0;
}

int Layer_load_param(void *_self, const ParamDict& pd)
{
    return 0;
}

int Layer_load_model(void *_self, const ModelBin& mb)
{
    return 0;
}

int Layer_forward_multi(void *_self, const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt)
{
    Layer *self = (Layer *)_self;

    if (!self->support_inplace)
        return -1;

    top_blobs = bottom_blobs;
    for (int i = 0; i < (int)top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blobs[i].clone(opt.blob_allocator);
        if (top_blobs[i].empty())
            return -100;
    }

    return self->forward_inplace_multi(self, top_blobs, opt);
}

int Layer_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    Layer *self = (Layer *)_self;

    if (!self->support_inplace)
        return -1;

    top_blob = bottom_blob.clone(opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    return self->forward_inplace(self, top_blob, opt);
}

int Layer_forward_inplace_multi(void *_self, std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/)
{
    return -1;
}

int Layer_forward_inplace(void *_self, Mat& /*bottom_top_blob*/, const Option& /*opt*/)
{
    return -1;
}

static const layer_registry_entry layer_registry[] =
{
#include "layer_registry.h"
};

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

#if NCNN_STRING
int layer_to_index(const char* type)
{
    for (int i=0; i<layer_registry_entry_count; i++)
    {
        if (strcmp(type, layer_registry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* create_layer(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer(index);
}
#endif // NCNN_STRING

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    Layer* layer = layer_creator();
    layer->typeindex = index;
    return layer;
}
