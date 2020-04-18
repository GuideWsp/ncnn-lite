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

/* class definition */
const cclass ClassBlob = {
    .size = sizeof(Blob),
    .ctor = Blob_ctor,
    .dtor = Blob_dtor
};

/* blob constructor */
void *Blob_ctor(void *_self, va_list *args)
{
    // get the pointer
    Blob *self = _self;

    // init producer and consumers
    self->producer = -1;
    vector_init(self->consumers);

    // init the matrix
    self->shape.data = 0;
    self->shape.refcount = 0;
    self->shape.elemsize = 0;
    self->shape.elempack = 0;
    self->shape.allocator = 0;
    self->shape.dims = 0;
    self->shape.w = 0;
    self->shape.h = 0;
    self->shape.c = 0;
    self->shape.cstep = 0;

    return self;
}

/* blob destructor */
void *Blob_dtor(void *_self)
{
    // get the pointer
    Blob *self = _self;

    // clear the consumers
    vector_destroy(self->consumers);

    return self;
}
