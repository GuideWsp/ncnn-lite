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

#include "blob.h"

void init_blob(Blob *blob)
{
    // init producer and consumers
    blob->producer = -1;
    vector_init(blob->consumers);

    // init the matrix
    blob->shape.data = 0;
    blob->shape.refcount = 0;
    blob->shape.elemsize = 0;
    blob->shape.elempack = 0;
    blob->shape.allocator = 0;
    blob->shape.dims = 0;
    blob->shape.w = 0;
    blob->shape.h = 0;
    blob->shape.c = 0;
    blob->shape.cstep = 0;
}

void uninit_blob(Blob *blob)
{
    vector_destroy(blob->consumers);
}