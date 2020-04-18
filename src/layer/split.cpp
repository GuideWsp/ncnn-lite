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

#include "split.h"

void *Split_ctor(void *_self, va_list *args)
{
    Layer *self = (Layer *)_self;

    self->one_blob_only = false;
    self->support_inplace = false;
    self->support_packing = true;
    self->support_bf16_storage = true;

    return _self;
}

int Split_forward(void *_self, const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt)
{
    const Mat& bottom_blob = bottom_blobs[0];
    for (size_t i=0; i<top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }

    return 0;
}
