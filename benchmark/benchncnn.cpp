// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <float.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h> // sleep()

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"

#include "cstl/utils.h"

int DataReaderFromEmpty_scan(void *handle, const char* format, void* p)
{
    return 0;
}

size_t DataReaderFromEmpty_read(void *handle, void* buf, size_t size)
{
    memset(buf, 0, size);
    return size;
}

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static UnlockedPoolAllocator g_blob_pool_allocator;
static PoolAllocator g_workspace_pool_allocator;

void benchmark(const char* comment, const Mat& _in, const Option& opt)
{
    Mat in = _in;
    in.fill(0.01f);

    Net net;

    net.opt = opt;

    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    net.load_param(parampath);

    DataReader dr = { .scan = DataReaderFromEmpty_scan, .read = DataReaderFromEmpty_read };
    net.load_model(NULL, dr);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
        sleep(10);
    }

    Mat out;

    // warm up
    for (int i=0; i<g_warmup_loop_count; i++)
    {
        Extractor ex = net.create_extractor();
        ex.input("data", in);
        ex.extract("output", out);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i=0; i<g_loop_count; i++)
    {
        double start = get_current_time();

        {
            Extractor ex = net.create_extractor();
            ex.input("data", in);
            ex.extract("output", out);
        }

        double end = get_current_time();

        double time = end - start;

        time_min = min(time_min, time);
        time_max = max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = get_cpu_count();
    int powersave = 0;
    int cooling_down = 1;

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        cooling_down = atoi(argv[4]);
    }

    g_enable_cooling_down = cooling_down != 0;

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    // default option
    Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;

    set_cpu_powersave(powersave);

    set_omp_dynamic(0);
    set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", get_cpu_powersave());
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);

    // run
    benchmark("squeezenet", Mat(227, 227, 3), opt);

    benchmark("squeezenet_int8", Mat(227, 227, 3), opt);

    benchmark("mobilenet", Mat(224, 224, 3), opt);

    benchmark("mobilenet_int8", Mat(224, 224, 3), opt);

    benchmark("mobilenet_v2", Mat(224, 224, 3), opt);

    // benchmark("mobilenet_v2_int8", Mat(224, 224, 3), opt);

    benchmark("mobilenet_v3", Mat(224, 224, 3), opt);

    benchmark("shufflenet", Mat(224, 224, 3), opt);

    benchmark("shufflenet_v2", Mat(224, 224, 3), opt);

    benchmark("mnasnet", Mat(224, 224, 3), opt);

    benchmark("proxylessnasnet", Mat(224, 224, 3), opt);

    benchmark("googlenet", Mat(224, 224, 3), opt);

    benchmark("googlenet_int8", Mat(224, 224, 3), opt);

    benchmark("resnet18", Mat(224, 224, 3), opt);

    benchmark("resnet18_int8", Mat(224, 224, 3), opt);

    benchmark("alexnet", Mat(227, 227, 3), opt);

    benchmark("vgg16", Mat(224, 224, 3), opt);

    benchmark("vgg16_int8", Mat(224, 224, 3), opt);

    benchmark("resnet50", Mat(224, 224, 3), opt);

    benchmark("resnet50_int8", Mat(224, 224, 3), opt);

    benchmark("squeezenet_ssd", Mat(300, 300, 3), opt);

    benchmark("squeezenet_ssd_int8", Mat(300, 300, 3), opt);

    benchmark("mobilenet_ssd", Mat(300, 300, 3), opt);

    benchmark("mobilenet_ssd_int8", Mat(300, 300, 3), opt);

    benchmark("mobilenet_yolo", Mat(416, 416, 3), opt);

    benchmark("mobilenetv2_yolov3", Mat(352, 352, 3), opt);

    return 0;
}
