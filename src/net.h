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

#ifndef NCNN_NET_H
#define NCNN_NET_H

#include <stdio.h>
#include <vector>
#include "platform.h"
#include "blob.h"
#include "layer.h"
#include "mat.h"
#include "option.h"

#include "cstl/vector.h"

struct DataReader;
struct Extractor;
struct Net
{
    // empty init
    Net();
    // clear and destroy
    ~Net();

    // option can be changed before loading
    Option opt;

#if NCNN_STRING
    // register custom layer by layer type name
    // return 0 if success
    int register_custom_layer(const char* type, layer_creator_func creator);
#endif // NCNN_STRING
    // register custom layer by layer type
    // return 0 if success
    int register_custom_layer(int index, layer_creator_func creator);

#if NCNN_STRING
    int load_param(const DataReader& dr);
#endif // NCNN_STRING

    int load_param_bin(const DataReader& dr);

    int load_model(const DataReader& dr);

#if NCNN_STDIO
#if NCNN_STRING
    // load network structure from plain param file
    // return 0 if success
    int load_param(FILE* fp);
    int load_param(const char* protopath);
    int load_param_mem(const char* mem);
#endif // NCNN_STRING
    // load network structure from binary param file
    // return 0 if success
    int load_param_bin(FILE* fp);
    int load_param_bin(const char* protopath);

    // load network weight data from model file
    // return 0 if success
    int load_model(FILE* fp);
    int load_model(const char* modelpath);
#endif // NCNN_STDIO

    // load network structure from external memory
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_param(const unsigned char* mem);

    // reference network weight data from external memory
    // weight data is not copied but referenced
    // so external memory should be retained when used
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_model(const unsigned char* mem);

#if __ANDROID_API__ >= 9
#if NCNN_STRING
    // convenient load network structure from android asset plain param file
    int load_param(AAsset* asset);
    int load_param(AAssetManager* mgr, const char* assetpath);
#endif // NCNN_STRING
    // convenient load network structure from android asset binary param file
    int load_param_bin(AAsset* asset);
    int load_param_bin(AAssetManager* mgr, const char* assetpath);

    // convenient load network weight data from android asset model file
    int load_model(AAsset* asset);
    int load_model(AAssetManager* mgr, const char* assetpath);
#endif // __ANDROID_API__ >= 9

    // unload network structure and weight data
    void clear();

    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, Option& opt) const;

    vector_def(Blob) blobs;
    vector_def(Layer*) layers;

    vector_def(layer_registry_entry) custom_layer_registry;
};

// construct an Extractor from network
extern Extractor create_extractor(Net *net);

#if NCNN_STRING

extern int find_blob_index_by_name(Net *net, const char* name);
extern int find_layer_index_by_name(Net *net, const char* name);
extern int custom_layer_to_index(Net *net, const char* type);
extern Layer* create_custom_layer_by_type(Net *net, const char* type);

#endif // NCNN_STRING

extern Layer* create_custom_layer_by_index(Net *net, int index);

// parse the structure of network
// fuse int8 op dequantize and quantize by requantize
int fuse_network(Net *net);

struct Extractor
{
    ~Extractor();

    // enable light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    void set_light_mode(bool enable);

    // set thread count for this extractor
    // this will overwrite the global setting
    // default count is system depended
    void set_num_threads(int num_threads);

    // set blob memory allocator
    void set_blob_allocator(Allocator* allocator);

    // set workspace memory allocator
    void set_workspace_allocator(Allocator* allocator);

#if NCNN_STRING
    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const Mat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, Mat& feat);
#endif // NCNN_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const Mat& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, Mat& feat);

    Extractor(const Net* net, size_t blob_count);

    const Net* net;
    std::vector<Mat> blob_mats;
    Option opt;
};

#endif // NCNN_NET_H
