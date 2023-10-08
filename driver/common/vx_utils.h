#pragma once

#include <cstdint>

uint64_t aligned_size(uint64_t size, uint64_t alignment);

bool is_aligned(uint64_t addr, uint64_t alignment);

#define CACHE_BLOCK_SIZE    64
// NOTE(hansung): This is changed to something more akin to be in a heap area
// for a CPU userspace program, since that works better with Chipyard's default
// memory mapping scheme (0x80000000 and above).  This gives us a pretty small
// space though.
#define ALLOC_BASE_ADDR     0xc0000000ul
#define LOCAL_MEM_SIZE      0x40000000ul     // 1 GB 
// #define ALLOC_BASE_ADDR     0x00000000
// #define LOCAL_MEM_SIZE      4294967296     // 4 GB 
#define DEVICE_MAX_ADDR     0xfffffffful
