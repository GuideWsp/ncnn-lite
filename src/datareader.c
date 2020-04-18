// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "datareader.h"
#include <string.h>
#include <stdlib.h>

#if NCNN_STDIO
#if NCNN_STRING
int DataReaderFromStdio_scan(const void *_self, const char* format, void* p)
{
    FILE *fp = (FILE *)((struct DataReader *)_self)->dr_handle;
    return fscanf(fp, format, p);
}
#endif // NCNN_STRING

size_t DataReaderFromStdio_read(const void *_self, void* buf, size_t size)
{
    FILE *fp = (FILE *)((struct DataReader *)_self)->dr_handle;
    return fread(buf, 1, size, fp);
}
#endif // NCNN_STDIO

#if NCNN_STRING
int DataReaderFromMemory_scan(const void *_self, const char* format, void* p)
{
    char **mem_ptr = (char **)((struct DataReader *)_self)->dr_handle;
    size_t fmtlen = strlen(format);

    char* format_with_n = (char *)malloc(fmtlen + 3);
    sprintf(format_with_n, "%s%%n", format);

    int nconsumed = 0;
    int nscan = sscanf(*mem_ptr, format_with_n, p, &nconsumed);
    *mem_ptr += nconsumed;

    free(format_with_n);

    return nconsumed > 0 ? nscan : 0;
}
#endif // NCNN_STRING

size_t DataReaderFromMemory_read(const void *_self, void* buf, size_t size)
{
    char **mem_ptr = (char **)((struct DataReader *)_self)->dr_handle;
    memcpy(buf, *mem_ptr, size);
    *mem_ptr += size;
    return size;
}
