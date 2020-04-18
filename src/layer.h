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

#ifndef NCNN_LAYER_H
#define NCNN_LAYER_H

#include <stdio.h>
#include <string>
#include <stdarg.h>
#include <vector>
#include <math.h>
#include "platform.h"
#include "mat.h"
#include "modelbin.h"
#include "option.h"
#include "paramdict.h"

#include "cstl/class.h"

struct Layer
{
    // cclass def
    cclass *clazz;

    // constructor and destructor
    // int (*ctor)(void *_self);
    // int (*dtor)(void *_self);

    // load layer specific parameter from parsed dict
    // return 0 if success
    int (*load_param)(void *_self, const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    int (*load_model)(void *_self, const ModelBin& mb);

    // layer implementation specific setup
    // return 0 if success
    int (*create_pipeline)(void *_self, const Option& opt);

    // layer implementation specific clean
    // return 0 if success
    int (*destroy_pipeline)(void *_self, const Option& opt);

    // one input and one output blob
    bool one_blob_only;

    // support inplace inference
    bool support_inplace;

    // accept input blob with packed storage
    bool support_packing;

    // accept bf16
    bool support_bf16_storage;

    // implement inference
    // return 0 if success
    int (*forward_multi)(void *_self, const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt);
    int (*forward)(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

    // implement inplace inference
    // return 0 if success
    int (*forward_inplace_multi)(void *_self, std::vector<Mat>& bottom_top_blobs, const Option& opt);
    int (*forward_inplace)(void *_self, Mat& bottom_top_blob, const Option& opt);

    // layer type index
    int typeindex;
#if NCNN_STRING
    // layer type name
    char type[256];
    // layer name
    char name[256];
#endif // NCNN_STRING
    // blob index which this layer needs as input
    std::vector<int> bottoms;
    // blob index which this layer produces as output
    std::vector<int> tops;
    // shape hint
    std::vector<Mat> bottom_shapes;
    std::vector<Mat> top_shapes;
};

// layer constructor
extern void *Layer_ctor(void *_self, va_list *args);

// layer destructor
extern void *Layer_dtor(void *_self);

int Layer_create_pipeline(void *_self, const Option& opt);

int Layer_destroy_pipeline(void *_self, const Option& opt);

int Layer_load_param(void *_self, const ParamDict& pd);

int Layer_load_model(void *_self, const ModelBin& mb);

int Layer_forward_multi(void *_self, const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt);

int Layer_forward(void *_self, const Mat& bottom_blob, Mat& top_blob, const Option& opt);

int Layer_forward_inplace_multi(void *_self, std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/);

int Layer_forward_inplace(void *_self, Mat& /*bottom_top_blob*/, const Option& /*opt*/);

// layer factory function
typedef Layer* (*layer_creator_func)();

struct layer_registry_entry
{
#if NCNN_STRING
    // layer type name
    const char* name;
#endif // NCNN_STRING
    // layer factory entry
    layer_creator_func creator;
};

#if NCNN_STRING
// get layer type from type name
int layer_to_index(const char* type);
// create layer from type name
Layer* create_layer(const char* type);
#endif // NCNN_STRING
// create layer from layer type
Layer* create_layer(int index);

#define DEFINE_LAYER_CREATOR(name) \
    const cclass cclazz_##name = {                      \
        .size = sizeof(name),                           \
        .ctor = name##_final_ctor,                      \
        .dtor = name##_final_dtor                       \
    };                                                  \
    Layer* name##_final_layer_creator() {               \
        return cnew((void *)&cclazz_##name,             \
                    name##_load_param,                  \
                    name##_load_model,                  \
                    name##_final_create_pipeline,       \
                    name##_final_destroy_pipeline,      \
                    name##_final_forward_multi,         \
                    name##_final_forward,               \
                    name##_final_forward_inplace_multi, \
                    name##_final_forward_inplace        \
        ); }

#endif // NCNN_LAYER_H
