// Leo is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 Leo <leo@nullptr.com.cn>. All rights reserved.
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


#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <time.h>

// get the current time
static inline char *get_curr_time()
{
    static char nowtime[20] = { 0 };
    time_t curr_time;
    time(&curr_time);
    strftime(nowtime, 20, "%Y-%m-%d %H:%M:%S", localtime(&curr_time));
    return nowtime;
}

// logging colors
#define FMT_COLOR_RED       "\e[0;31m"
#define FMT_COLOR_YELLOW    "\e[0;33m"
#define FMT_COLOR_CYAN      "\e[0;36m"
#define FMT_COLOR_END       "\e[0m"

// logging prefix
#define LOG_PREFIX_ERR      FMT_COLOR_RED "[ERROR]" FMT_COLOR_END
#define LOG_PREFIX_INFO     "[INFO]"
#define LOG_PREFIX_DBG      "[DEBUG]"
#define LOG_PREFIX_TRACE    "[TRACE]"

// logging functions
#ifdef DEBUG
#define log_info(fmt, ...) printf(LOG_PREFIX_INFO "[%s] %s(%d): " fmt "\n", get_curr_time(), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define log_err(fmt, ...) printf(LOG_PREFIX_ERR "[%s] %s(%d): " fmt "\n", get_curr_time(), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define log_dbg(fmt, ...) printf(LOG_PREFIX_DBG "[%s] %s(%d): " fmt "\n", get_curr_time(), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define log_info(fmt, ...) printf(LOG_PREFIX_INFO "[%s] " fmt "\n", get_curr_time(), ##__VA_ARGS__)
#define log_err(fmt, ...) printf(LOG_PREFIX_ERR "[%s] " fmt "\n", get_curr_time(), ##__VA_ARGS__)
#define log_dbg(...)
#endif

// trace functions
#ifdef TRACE
#define log_trace(fmt, ...) printf(LOG_PREFIX_TRACE "[%s] %s(%d): " fmt "\n", get_curr_time(), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define log_trace(...)
#endif

#endif
