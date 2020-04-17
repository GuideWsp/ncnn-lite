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

#include "allocator.h"

#include <stdio.h>
#include <algorithm>

Allocator::~Allocator() 
{

}

PoolAllocator::PoolAllocator()
{
    size_compare_ratio = 192;// 0.75f * 256

    // initialize mutex
    pthread_mutex_init(&budgets_lock, 0);
    pthread_mutex_init(&payouts_lock, 0);
}

PoolAllocator::~PoolAllocator()
{
    clear();

    if (!payouts.empty())
    {
        fprintf(stderr, "FATAL ERROR! pool allocator destroyed too early\n");
        std::list< std::pair<size_t, void*> >::iterator it = payouts.begin();
        for (; it != payouts.end(); it++)
        {
            void* ptr = it->second;
            fprintf(stderr, "%p still in use\n", ptr);
        }
    }

    // uninitialize mutex
    pthread_mutex_destroy(&budgets_lock);
    pthread_mutex_destroy(&payouts_lock);
}

void PoolAllocator::clear()
{
    pthread_mutex_lock(&budgets_lock);

    std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        void* ptr = it->second;
        fastFree(ptr);
    }
    budgets.clear();

    pthread_mutex_unlock(&budgets_lock);
}

void PoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        fprintf(stderr, "invalid size compare ratio %f\n", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

void* PoolAllocator::fastMalloc(size_t size)
{
    pthread_mutex_lock(&budgets_lock);

    // find free budget
    std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            budgets.erase(it);

            pthread_mutex_unlock(&budgets_lock);

            pthread_mutex_lock(&payouts_lock);

            payouts.push_back(std::make_pair(bs, ptr));

            pthread_mutex_unlock(&payouts_lock);

            return ptr;
        }
    }

    pthread_mutex_unlock(&budgets_lock);

    // new
    void* ptr = fastMalloc(size);

    pthread_mutex_lock(&payouts_lock);

    payouts.push_back(std::make_pair(size, ptr));

    pthread_mutex_unlock(&payouts_lock);

    return ptr;
}

void PoolAllocator::fastFree(void* ptr)
{
    pthread_mutex_lock(&payouts_lock);

    // return to budgets
    std::list< std::pair<size_t, void*> >::iterator it = payouts.begin();
    for (; it != payouts.end(); it++)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            payouts.erase(it);

            pthread_mutex_unlock(&payouts_lock);

            pthread_mutex_lock(&budgets_lock);

            budgets.push_back(std::make_pair(size, ptr));

            pthread_mutex_unlock(&budgets_lock);

            return;
        }
    }

    pthread_mutex_unlock(&payouts_lock);

    fprintf(stderr, "FATAL ERROR! pool allocator get wild %p\n", ptr);
    fastFree(ptr);
}

UnlockedPoolAllocator::UnlockedPoolAllocator()
{
    size_compare_ratio = 192;// 0.75f * 256
}

UnlockedPoolAllocator::~UnlockedPoolAllocator()
{
    clear();

    if (!payouts.empty())
    {
        fprintf(stderr, "FATAL ERROR! unlocked pool allocator destroyed too early\n");
        std::list< std::pair<size_t, void*> >::iterator it = payouts.begin();
        for (; it != payouts.end(); it++)
        {
            void* ptr = it->second;
            fprintf(stderr, "%p still in use\n", ptr);
        }
    }
}

void UnlockedPoolAllocator::clear()
{
    std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        void* ptr = it->second;
        fastFree(ptr);
    }
    budgets.clear();
}

void UnlockedPoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        fprintf(stderr, "invalid size compare ratio %f\n", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

void* UnlockedPoolAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            budgets.erase(it);

            payouts.push_back(std::make_pair(bs, ptr));

            return ptr;
        }
    }

    // new
    void* ptr = fastMalloc(size);

    payouts.push_back(std::make_pair(size, ptr));

    return ptr;
}

void UnlockedPoolAllocator::fastFree(void* ptr)
{
    // return to budgets
    std::list< std::pair<size_t, void*> >::iterator it = payouts.begin();
    for (; it != payouts.end(); it++)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            payouts.erase(it);

            budgets.push_back(std::make_pair(size, ptr));

            return;
        }
    }

    fprintf(stderr, "FATAL ERROR! unlocked pool allocator get wild %p\n", ptr);
    fastFree(ptr);
}
