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

#include "net.h"
#include "layer_type.h"
#include "datareader.h"
#include "modelbin.h"
#include "paramdict.h"
#include "convolution.h"
#include "convolutiondepthwise.h"
#include "relu.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "cstl/strings.h"

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

Net::Net()
{
    vector_init_ctor_dtor(blobs, Blob_ctor, Blob_dtor);
}

Net::~Net()
{
    clear();
    vector_destroy(blobs);
}

#if NCNN_STRING
int Net::register_custom_layer(const char* type, layer_creator_func creator)
{
    int typeindex = layer_to_index(type);
    if (typeindex != -1)
    {
        fprintf(stderr, "can not register build-in layer type %s\n", type);
        return -1;
    }

    int custom_index = custom_layer_to_index(type);
    if (custom_index == -1)
    {
        struct layer_registry_entry entry = { type, creator };
        custom_layer_registry.push_back(entry);
    }
    else
    {
        fprintf(stderr, "overwrite existing custom layer type %s\n", type);
        custom_layer_registry[custom_index].name = type;
        custom_layer_registry[custom_index].creator = creator;
    }

    return 0;
}
#endif // NCNN_STRING

int Net::register_custom_layer(int index, layer_creator_func creator)
{
    int custom_index = index & ~CustomBit;
    if (index == custom_index)
    {
        fprintf(stderr, "can not register build-in layer index %d\n", custom_index);
        return -1;
    }

    if ((int)custom_layer_registry.size() <= custom_index)
    {
#if NCNN_STRING
        struct layer_registry_entry dummy = { "", 0 };
#else
        struct layer_registry_entry dummy = { 0 };
#endif // NCNN_STRING
        custom_layer_registry.resize(custom_index + 1, dummy);
    }

    if (custom_layer_registry[custom_index].creator)
    {
        fprintf(stderr, "overwrite existing custom layer index %d\n", custom_index);
    }

    custom_layer_registry[custom_index].creator = creator;
    return 0;
}

#if NCNN_STRING
int Net::load_param(const DataReader& dr)
{
#define SCAN_VALUE(fmt, v) \
    if (dr.scan(&dr, fmt, &v) != 1) \
    { \
        fprintf(stderr, "parse " #v " failed\n"); \
        return -1; \
    }

    int magic = 0;
    SCAN_VALUE("%d", magic)
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        fprintf(stderr, "invalid layer_count or blob_count\n");
        return -1;
    }

    layers.resize((size_t)layer_count);
    vector_resize(blobs, blob_count);

    ParamDict pd;

    int blob_index = 0;
    for (int i=0; i<layer_count; i++)
    {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)

        Layer* layer = create_layer(layer_type);
        if (!layer)
        {
            layer = create_custom_layer(layer_type);
        }
        if (!layer)
        {
            fprintf(stderr, "layer %s not exists or registered\n", layer_type);
            clear();
            return -1;
        }

        strcpy_s(layer->type, 256, layer_type);
        strcpy_s(layer->name, 256, layer_name);
//         fprintf(stderr, "new layer %d %s\n", i, layer_name);

        layer->bottoms.resize(bottom_count);

        for (int j=0; j<bottom_count; j++)
        {
            char bottom_name[256];
            SCAN_VALUE("%255s", bottom_name)

            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            if (bottom_blob_index == -1)
            {
                Blob& blob = vector_get(blobs, blob_index);

                bottom_blob_index = blob_index;

                strcpy_s(blob.name, 256, bottom_name);
//                 fprintf(stderr, "new blob %s\n", bottom_name);

                blob_index++;
            }

            Blob& blob = vector_get(blobs, bottom_blob_index);

            vector_pushback(blob.consumers, i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            Blob& blob = vector_get(blobs, blob_index);

            char blob_name[256];
            SCAN_VALUE("%255s", blob_name)

            strcpy_s(blob.name, 256, blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param(dr);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        // pull out top shape hints
        Mat shape_hints = pd.get(30, Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j=0; j<top_count; j++)
            {
                Blob& blob = vector_get(blobs, layer->tops[j]);

                int dims = psh[0];
                if (dims == 1)
                {
                    blob.shape = Mat(psh[1], (void*)0, 4u, 1);
                }
                if (dims == 2)
                {
                    blob.shape = Mat(psh[1], psh[2], (void*)0, 4u, 1);
                }
                if (dims == 3)
                {
                    blob.shape = Mat(psh[1], psh[2], psh[3], (void*)0, 4u, 1);
                }

                psh += 4;
            }
        }

        // set bottom and top shape hints
        layer->bottom_shapes.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            layer->bottom_shapes[j] = vector_get(blobs, layer->bottoms[j]).shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            layer->top_shapes[j] = vector_get(blobs, layer->tops[j]).shape;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

#undef SCAN_VALUE
    return 0;
}
#endif // NCNN_STRING

int Net::load_param_bin(const DataReader& dr)
{
#define READ_VALUE(buf) \
    if (dr.read(&dr, &buf, sizeof(buf)) != sizeof(buf)) \
    { \
        fprintf(stderr, "read " #buf " failed\n"); \
        return -1; \
    }

    int magic = 0;
    READ_VALUE(magic)
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    int layer_count = 0;
    int blob_count = 0;
    READ_VALUE(layer_count)
    READ_VALUE(blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        fprintf(stderr, "invalid layer_count or blob_count\n");
        return -1;
    }

    layers.resize(layer_count);
    vector_resize(blobs, blob_count);

    ParamDict pd;

    for (int i=0; i<layer_count; i++)
    {
        int typeindex;
        int bottom_count;
        int top_count;
        READ_VALUE(typeindex)
        READ_VALUE(bottom_count)
        READ_VALUE(top_count)

        Layer* layer = create_layer(typeindex);
        if (!layer)
        {
            int custom_index = typeindex & ~CustomBit;
            layer = create_custom_layer(custom_index);
        }
        if (!layer)
        {
            fprintf(stderr, "layer %d not exists or registered\n", typeindex);
            clear();
            return -1;
        }

//         strcpy_s(layer->type, 256, layer_type);
//         strcpy_s(layer->name, 256, layer_name);
//         fprintf(stderr, "new layer %d\n", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index;
            READ_VALUE(bottom_blob_index)

            Blob& blob = vector_get(blobs, bottom_blob_index);

            vector_pushback(blob.consumers, i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index;
            READ_VALUE(top_blob_index)

            Blob& blob = vector_get(blobs, top_blob_index);

//             strcpy_s(blob.name, 256, blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = top_blob_index;
        }

        // layer specific params
        int pdlr = pd.load_param_bin(dr);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        // pull out top blob shape hints
        Mat shape_hints = pd.get(30, Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j=0; j<top_count; j++)
            {
                Blob& blob = vector_get(blobs, layer->tops[j]);

                int dims = psh[0];
                if (dims == 1)
                {
                    blob.shape = Mat(psh[1], (void*)0, 4u, 1);
                }
                if (dims == 2)
                {
                    blob.shape = Mat(psh[1], psh[2], (void*)0, 4u, 1);
                }
                if (dims == 3)
                {
                    blob.shape = Mat(psh[1], psh[2], psh[3], (void*)0, 4u, 1);
                }

                psh += 4;
            }
        }

        // set bottom and top shape hints
        layer->bottom_shapes.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            layer->bottom_shapes[j] = vector_get(blobs, layer->bottoms[j]).shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            layer->top_shapes[j] = vector_get(blobs, layer->tops[j]).shape;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

#undef READ_VALUE
    return 0;
}

int Net::load_model(const DataReader& dr)
{
    if (layers.empty())
    {
        fprintf(stderr, "network graph not ready\n");
        return -1;
    }

    // load file
    int ret = 0;

    ModelBinFromDataReader mb(dr);
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
            fprintf(stderr, "load_model error at layer %d, parameter file has inconsistent content.\n", (int)i);
            ret = -1;
            break;
        }

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
            fprintf(stderr, "layer load_model %d failed\n", (int)i);
            ret = -1;
            break;
        }
    }

    fuse_network();

    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
            fprintf(stderr, "load_model error at layer %d, parameter file has inconsistent content.\n", (int)i);
            ret = -1;
            break;
        }

        int cret = layer->create_pipeline(opt);
        if (cret != 0)
        {
            fprintf(stderr, "layer create_pipeline %d failed\n", (int)i);
            ret = -1;
            break;
        }
    }

    return ret;
}

#if NCNN_STDIO
#if NCNN_STRING
int Net::load_param(FILE* fp)
{
    DataReader dr = createDataReaderFromStdio(fp);
    return load_param(dr);
}

int Net::load_param_mem(const char* _mem)
{
    const unsigned char* mem = (const unsigned char*)_mem;
    DataReader dr = createDataReaderFromMemory(&mem);
    return load_param(dr);
}

int Net::load_param(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", protopath);
        return -1;
    }

    int ret = load_param(fp);
    fclose(fp);
    return ret;
}
#endif // NCNN_STRING

int Net::load_param_bin(FILE* fp)
{
    DataReader dr = createDataReaderFromStdio(fp);
    return load_param_bin(dr);
}

int Net::load_param_bin(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", protopath);
        return -1;
    }

    int ret = load_param_bin(fp);
    fclose(fp);
    return ret;
}

int Net::load_model(FILE* fp)
{
    DataReader dr = createDataReaderFromStdio(fp);
    return load_model(dr);
}

int Net::load_model(const char* modelpath)
{
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", modelpath);
        return -1;
    }

    int ret = load_model(fp);
    fclose(fp);
    return ret;
}
#endif // NCNN_STDIO

int Net::load_param(const unsigned char* _mem)
{
    const unsigned char* mem = _mem;
    DataReader dr = createDataReaderFromMemory(&mem);
    load_param_bin(dr);
    return static_cast<int>(mem - _mem);
}

int Net::load_model(const unsigned char* _mem)
{
    const unsigned char* mem = _mem;
    DataReader dr = createDataReaderFromMemory(&mem);
    load_model(dr);
    return static_cast<int>(mem - _mem);
}

int Net::fuse_network()
{
    // set the int8 op fusion:requantize
#if NCNN_STRING && NCNN_REQUANT    
    // fprintf(stderr, "Test op fusion to int8 implement:\n");
    // parse the network whether is a quantization model
    bool net_quantized = false;
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];
        if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise")
        {
            if (layer->type == "Convolution" && (((Convolution*)layer)->weight_data.elemsize != 1u))
                continue;
            if (layer->type == "ConvolutionDepthWise" && (((ConvolutionDepthWise*)layer)->weight_data.elemsize != 1u))
                continue;    
            net_quantized = true;
        }
    }

    if (net_quantized == false)
        return 0;

    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise")
        {
            if (layer->type == "Convolution" && (((Convolution*)layer)->weight_data.elemsize != 1u))
                continue;
            if (layer->type == "ConvolutionDepthWise" && (((ConvolutionDepthWise*)layer)->weight_data.elemsize != 1u))
                continue;

            for (size_t n=0; n<vector_size(blobs[layer->tops[0]].consumers); n++)
            {
                int layer_next_index = vector_get(blobs[layer->tops[0]].consumers, n);
                Layer* layer_next = layers[layer_next_index];

                if (layer_next->type == "Convolution" || layer_next->type == "ConvolutionDepthWise")
                {
                    if (layer_next->type == "Convolution" && ((Convolution*)layer_next)->weight_data.elemsize != 1u)
                        continue;
                    if (layer_next->type == "ConvolutionDepthWise" && ((ConvolutionDepthWise*)layer_next)->weight_data.elemsize != 1u)
                        continue;    

                    // fprintf(stderr, "%s, %s\n", layer->name.c_str(), layer_next->name.c_str());
                    if (layer->type == "Convolution" && layer_next->type == "Convolution")
                    {
                        ((Convolution*)layer)->use_int8_requantize = true;
                        ((Convolution*)layer)->top_blob_int8_scale = ((Convolution*)layer_next)->bottom_blob_int8_scale;
                    }
                    else if (layer->type == "ConvolutionDepthWise" && layer_next->type == "Convolution")
                    {
                        ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                        ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((Convolution*)layer_next)->bottom_blob_int8_scale;
                    }
                    else if (layer->type == "Convolution" && layer_next->type == "ConvolutionDepthWise")
                    {
                        ((Convolution*)layer)->use_int8_requantize = true;
                        ((Convolution*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next)->bottom_blob_int8_scales[0];
                    }
                    else
                    {
                        ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                        ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next)->bottom_blob_int8_scales[0];
                    }
                }                  
                else if (layer_next->type == "ReLU")
                {
                    int layer_next_2_index = vector_get(blobs[layer_next->tops[0]].consumers, 0);
                    Layer* layer_next_2 = layers[layer_next_2_index];

                    if (layer_next_2->type == "Convolution" || layer_next_2->type == "ConvolutionDepthWise")
                    {
                        if (layer_next_2->type == "Convolution" && ((Convolution*)layer_next_2)->weight_data.elemsize != 1u)
                            continue;
                        if (layer_next_2->type == "ConvolutionDepthWise" && ((ConvolutionDepthWise*)layer_next_2)->weight_data.elemsize != 1u)
                            continue;    

//                         fprintf(stderr, "%s, %s, %s\n", layer->name.c_str(), layer_next->name.c_str(), layer_next_2->name.c_str());
                        if (layer->type == "Convolution" && layer_next_2->type == "Convolution")
                        {
                            ((Convolution*)layer)->use_int8_requantize = true;
                            ((Convolution*)layer)->top_blob_int8_scale = ((Convolution*)layer_next_2)->bottom_blob_int8_scale;
                        }
                        else if (layer->type == "ConvolutionDepthWise" && layer_next_2->type == "Convolution")
                        {
                            ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                            ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((Convolution*)layer_next_2)->bottom_blob_int8_scale;
                        }
                        else if (layer->type == "Convolution" && layer_next_2->type == "ConvolutionDepthWise")
                        {
                            ((Convolution*)layer)->use_int8_requantize = true;
                            ((Convolution*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next_2)->bottom_blob_int8_scales[0];
                        }
                        else
                        {
                            ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                            ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next_2)->bottom_blob_int8_scales[0];
                        }
                    }
                    else if (layer_next_2->type == "Split")
                    {
                        bool all_conv = true;
                        for (size_t i=0; i<layer_next_2->tops.size(); i++)
                        {
                            int layer_next_3_index = vector_get(blobs[layer_next_2->tops[i]].consumers, 0);
                            if (layers[layer_next_3_index]->type != "Convolution" && layers[layer_next_3_index]->type != "ConvolutionDepthWise" && layers[layer_next_3_index]->type != "PriorBox" )
                            {
                                // fprintf(stderr, "%s, %s, %s, %s\n", layer->name.c_str(), layer_next->name.c_str(), layer_next_2->name.c_str(), layers[layer_next_3_index]->name.c_str());
                                all_conv = false;
                            }
                        }

                        if (all_conv == true && layer_next_2->tops.size() >= size_t(2))
                        {
                            // fprintf(stderr, "%s, %s, %s, ", layer->name.c_str(), layer_next->name.c_str(), layer_next_2->name.c_str());
                            for (size_t i=0; i<layer_next_2->tops.size(); i++)
                            {
                                int layer_next_3_index = vector_get(blobs[layer_next_2->tops[i]].consumers, 0);
                                Layer* layer_next_3 = layers[layer_next_3_index];

                                // fprintf(stderr, "%s, ", layer_next_3->name.c_str());
                                if (layer_next_3->type == "Convolution")
                                {
                                    ((Convolution*)layer)->top_blob_int8_scale = ((Convolution*)layer_next_3)->bottom_blob_int8_scale; 
                                }    
                            }

                            ((Convolution*)layer)->use_int8_requantize = true;
                            // fprintf(stderr, "\n");
                        }
                    }
                    else
                    {
                        // fprintf(stderr, "%s, %s\n", layer->name.c_str(), layer_next->name.c_str());
                    }
                }
                else if (layer_next->type == "Pooling")
                {
                    // ToDo
                }
                else
                {
                    // fprintf(stderr, "%s\n", layer->name.c_str());
                }                  
            }
        }
    }
#endif
    return 0;
}

void Net::clear()
{
    vector_clear(blobs);
    for (size_t i=0; i<layers.size(); i++)
    {
        int dret = layers[i]->destroy_pipeline(opt);
        if (dret != 0)
        {
            fprintf(stderr, "layer destroy_pipeline failed\n");
            // ignore anyway
        }

        delete layers[i];
    }
    layers.clear();
}

Extractor Net::create_extractor() const
{
    return Extractor(this, vector_size(blobs));
}

#if NCNN_STRING
int Net::find_blob_index_by_name(const char* name) const
{
    for (size_t i=0; i<vector_size(blobs); i++)
    {
        const Blob& blob = vector_get(blobs, i);
        if (strcmp(blob.name, name) == 0)
        {
            return static_cast<int>(i);
        }
    }

    fprintf(stderr, "find_blob_index_by_name %s failed\n", name);
    return -1;
}

int Net::find_layer_index_by_name(const char* name) const
{
    for (size_t i=0; i<layers.size(); i++)
    {
        const Layer* layer = layers[i];
        if (layer->name == name)
        {
            return static_cast<int>(i);
        }
    }

    fprintf(stderr, "find_layer_index_by_name %s failed\n", name);
    return -1;
}

int Net::custom_layer_to_index(const char* type)
{
    const size_t custom_layer_registry_entry_count = custom_layer_registry.size();
    for (size_t i=0; i<custom_layer_registry_entry_count; i++)
    {
        if (strcmp(type, custom_layer_registry[i].name) == 0)
            return static_cast<int>(i);
    }

    return -1;
}

Layer* Net::create_custom_layer(const char* type)
{
    int index = custom_layer_to_index(type);
    if (index == -1)
        return 0;

    return create_custom_layer(index);
}
#endif // NCNN_STRING

Layer* Net::create_custom_layer(int index)
{
    const size_t custom_layer_registry_entry_count = custom_layer_registry.size();
    if (index < 0 || static_cast<unsigned int>(index) >= custom_layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = custom_layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    return layer_creator();
}

int Net::forward_layer(int layer_index, std::vector<Mat>& blob_mats, Option& opt) const
{
    const Layer* layer = layers[layer_index];

//     fprintf(stderr, "forward_layer %d %s\n", layer_index, layer->name.c_str());

    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        if (blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(vector_get(blobs, bottom_blob_index).producer, blob_mats, opt);
            if (ret != 0)
                return ret;
        }

        Mat bottom_blob = blob_mats[bottom_blob_index];

        if (opt.lightmode)
        {
            // delete after taken in light mode
            blob_mats[bottom_blob_index].release();
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob.refcount != 1)
            {
                bottom_blob = bottom_blob.clone();
            }
        }

        if (opt.use_bf16_storage)
        {
            if (bottom_blob.elemsize / bottom_blob.elempack == 4u && layer->support_bf16_storage)
            {
                Mat bottom_blob_bf16;
                cast_float32_to_bfloat16(bottom_blob, bottom_blob_bf16, opt);
                bottom_blob = bottom_blob_bf16;
            }
            if (bottom_blob.elemsize / bottom_blob.elempack == 2u && !layer->support_bf16_storage)
            {
                Mat bottom_blob_fp32;
                cast_bfloat16_to_float32(bottom_blob, bottom_blob_fp32, opt);
                bottom_blob = bottom_blob_fp32;
            }
        }

        if (opt.use_packing_layout)
        {
            int elempack = layer->support_packing ? 4 : 1;

            Mat bottom_blob_packed;
            convert_packing(bottom_blob, bottom_blob_packed, elempack, opt);
            bottom_blob = bottom_blob_packed;
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            Mat& bottom_top_blob = bottom_blob;
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward_inplace(bottom_top_blob, opt);
            double end = get_current_time();
            benchmark(layer, bottom_top_blob, bottom_top_blob, start, end);
#else
            int ret = layer->forward_inplace(bottom_top_blob, opt);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = bottom_top_blob;
        }
        else
        {
            Mat top_blob;
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward(bottom_blob, top_blob, opt);
            double end = get_current_time();
            benchmark(layer, bottom_blob, top_blob, start, end);
#else
            int ret = layer->forward(bottom_blob, top_blob, opt);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = top_blob;
        }

    }
    else
    {
        // load bottom blobs
        std::vector<Mat> bottom_blobs(layer->bottoms.size());
        for (size_t i=0; i<layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            if (blob_mats[bottom_blob_index].dims == 0)
            {
                int ret = forward_layer(vector_get(blobs, bottom_blob_index).producer, blob_mats, opt);
                if (ret != 0)
                    return ret;
            }

            bottom_blobs[i] = blob_mats[bottom_blob_index];

            if (opt.lightmode)
            {
                // delete after taken in light mode
                blob_mats[bottom_blob_index].release();
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blobs[i].refcount != 1)
                {
                    bottom_blobs[i] = bottom_blobs[i].clone();
                }
            }

            if (opt.use_bf16_storage)
            {
                if (bottom_blobs[i].elemsize / bottom_blobs[i].elempack == 4u && layer->support_bf16_storage)
                {
                    Mat bottom_blob_bf16;
                    cast_float32_to_bfloat16(bottom_blobs[i], bottom_blob_bf16, opt);
                    bottom_blobs[i] = bottom_blob_bf16;
                }
                if (bottom_blobs[i].elemsize / bottom_blobs[i].elempack == 2u && !layer->support_bf16_storage)
                {
                    Mat bottom_blob_fp32;
                    cast_bfloat16_to_float32(bottom_blobs[i], bottom_blob_fp32, opt);
                    bottom_blobs[i] = bottom_blob_fp32;
                }
            }

            if (opt.use_packing_layout)
            {
                int elempack = layer->support_packing ? 4 : 1;

                Mat bottom_blob_packed;
                convert_packing(bottom_blobs[i], bottom_blob_packed, elempack, opt);
                bottom_blobs[i] = bottom_blob_packed;
            }
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            std::vector<Mat>& bottom_top_blobs = bottom_blobs;
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward_inplace(bottom_top_blobs, opt);
            double end = get_current_time();
            benchmark(layer, start, end);
#else
            int ret = layer->forward_inplace(bottom_top_blobs, opt);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i=0; i<layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<Mat> top_blobs(layer->tops.size());
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward(bottom_blobs, top_blobs, opt);
            double end = get_current_time();
            benchmark(layer, start, end);
#else
            int ret = layer->forward(bottom_blobs, top_blobs, opt);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i=0; i<layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = top_blobs[i];
            }
        }
    }

//     fprintf(stderr, "forward_layer %d %s done\n", layer_index, layer->name.c_str());
//     const Mat& blob = blob_mats[layer->tops[0]];
//     fprintf(stderr, "[%-2d %-16s %-16s]  %d    blobs count = %-3d   size = %-3d x %-3d\n", layer_index, layer->type.c_str(), layer->name.c_str(), layer->tops[0], blob.c, blob.h, blob.w);

    return 0;
}

Extractor::Extractor(const Net* _net, size_t blob_count) : net(_net)
{
    blob_mats.resize(blob_count);
    opt = net->opt;
}

Extractor::~Extractor()
{
    blob_mats.clear();
}

void Extractor::set_light_mode(bool enable)
{
    opt.lightmode = enable;
}

void Extractor::set_num_threads(int num_threads)
{
    opt.num_threads = num_threads;
}

void Extractor::set_blob_allocator(Allocator* allocator)
{
    opt.blob_allocator = allocator;
}

void Extractor::set_workspace_allocator(Allocator* allocator)
{
    opt.workspace_allocator = allocator;
}

#if NCNN_STRING
int Extractor::input(const char* blob_name, const Mat& in)
{
    int blob_index = net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    return input(blob_index, in);
}

int Extractor::extract(const char* blob_name, Mat& feat)
{
    int blob_index = net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    return extract(blob_index, feat);
}
#endif // NCNN_STRING

int Extractor::input(int blob_index, const Mat& in)
{
    if (blob_index < 0 || blob_index >= (int)blob_mats.size())
        return -1;

    blob_mats[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, Mat& feat)
{
    if (blob_index < 0 || blob_index >= (int)blob_mats.size())
        return -1;

    int ret = 0;

    if (blob_mats[blob_index].dims == 0)
    {
        int layer_index = vector_get(net->blobs, blob_index).producer;
    }

    feat = blob_mats[blob_index];

    if (opt.use_packing_layout)
    {
        Mat bottom_blob_unpacked;
        convert_packing(feat, bottom_blob_unpacked, 1, opt);
        feat = bottom_blob_unpacked;
    }

    return ret;
}
