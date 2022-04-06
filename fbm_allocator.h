/* Copyright 2021 Francis Birck Moreira GPL */

#include<string.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>

#ifndef _FBM_ALLOCATOR_H_
    #define _FBM_ALLOCATOR_H_
#endif  // _FBM_ALLOCATOR_H



typedef struct mempool
{
    uint64_t * base;  // pointer to the start of the allocated memory
    uint64_t *** alloc_tag;  // vector containing the addresses of the variables that reference the allocated memory. We must change these pointers whenever we reallocate the memory pool.
    uint64_t chunklen;  // length of a chunk, in number of uint64_t positions
    uint64_t chunks;  // number of chunks
    uint64_t cur_pop;  // variable to track the number of allocated chunks. Faster way to find out if we need to reallocate
} mempool;

void
initpool(mempool * pool, uint64_t chunk_len, uint64_t chunks)
{
    // each bit in pool->bm_users indicates a chunk.
    // a chunk has chunk_len bits. so it allocates (chunk_len >> 6 + 1) (uint64_t)
    // since we are using avx512, we need to be aligned and the chunks must be multiples of 64B. so we actually want to allocate a bit extra, if it comes to that. But ideally the pointer is uint64_t for other purposes.
    const uint64_t avx_align = 64;
    const uint64_t log64 = 0x6ULL;
    const uint64_t base = 0x8ULL;

    uint64_t leftover = base - ((chunk_len >> log64) & 0x7ULL);  // & 7 means % 8.. we want to allocate aligned and multiple to/of 64~B, so we make a check of whether we have a multiple of 8 uint64_t. If we don't, add the leftover.
    pool->chunks = chunks;  // the number of chunks, plain and simple.
    pool->chunklen = (chunk_len >> (0x6ULL)) + leftover;  // alloc multiple of 64~B (8 uint64_t, 1 m512 for avx)
    pool->base = (uint64_t*)aligned_alloc(avx_align, sizeof(uint64_t) * pool->chunklen * pool->chunks);  // allocate #chunks of #chunk_len
    pool->alloc_tag = (uint64_t***)aligned_alloc(avx_align, sizeof(uint64_t**) * pool->chunks);  // array with addresses of the variables that hold references to the pool. We need to update these guys on reallocations and frees.
    memset(pool->alloc_tag, 0, pool->chunks * sizeof(uint64_t***));  // 0 all refs to backers
    memset(pool->base, 0, (pool->chunklen * pool->chunks * sizeof(uint64_t)));  // chunklen counts number of uint64_t (8~B).
    pool->cur_pop = 0;
}


// free the memory pool
void
destroypool(mempool * pool)
{
    if (pool->base != NULL)
        free(pool->base);
    if (pool->alloc_tag != NULL)
        free(pool->alloc_tag);
    pool->chunklen = 0;
    pool->chunks = 0;
}


// allocate a chunk from pool
uint32_t
allocpool(mempool * pool, uint64_t ** allocator)
{
  // perf complains: most time in allocpool is NULL check.
  for (uint64_t i = 0; i < pool->chunks; i++)
  {
    if (pool->alloc_tag[i] == NULL)  // free slot
    {
        pool->alloc_tag[i] = allocator;  // save the allocator var address
        *allocator = pool->base + i*pool->chunklen;  // make it point to chunk.
        memset(*allocator, 0, (pool->chunklen * sizeof(uint64_t)));  // reset upon allocation to new use.
        pool->cur_pop++;  // allocated someone, keep track of that
        if (pool->cur_pop >= (pool->chunks))  // near limit, time to reallocate
        {
          uint64_t * prevbase = pool->base;  // save previous base
          // double size of allocation
          pool->chunks <<= 0x1ULL;
          // reallocate
          pool->base = (uint64_t*)realloc(pool->base,
                                          sizeof(uint64_t) * pool->chunklen * pool->chunks);
          if (pool->base == NULL)  // failed to reallocate
          {
              printf("could not realloc, getting the hell out of here. tried size %lu\n", 
                sizeof(uint64_t) * pool->chunklen * pool->chunks);
              free(pool->alloc_tag);
              exit(0);
          }
          if (((uint64_t)pool->base & 0x3fULL) != 0)  // check misalignment.
          {
            uint64_t asize = sizeof(uint64_t)*pool->chunklen*pool->chunks;  // total size we want to allocate in bytes.
            uint64_t * aux = (uint64_t*)aligned_alloc(64, asize);  // grab aligned memory
            if (aux == NULL)
            {
              printf("aligned alloc failed\n");	
              exit(0);
            }
            memcpy(aux, pool->base, asize >> 0x1ULL);  // copy the entire previous pool to half the new memory.
            memset(aux + (asize >> 0x4ULL), 0, asize >> 0x1ULL);  // zero the next half, since we did not allocate anything yet. We divide asize by 16 because we want half the number of bytes, in portions of 8~bytes (since we are incrementing an uint64_t ptr)
            free(pool->base);  // deallocate previous memory
            pool->base = aux;  // take responsibility of new block
          }
          // adjust backreferences.
          int64_t diff = (int64_t)((uint64_t)pool->base - (uint64_t)prevbase);
          pool->alloc_tag = (uint64_t***)realloc(pool->alloc_tag,
                                                    sizeof(uint64_t***) * pool->chunks);
          for (uint32_t alloci = 0; alloci < (pool->chunks >> 0x1ULL); alloci++)
          {
            if ((*(pool->alloc_tag[alloci]) != NULL))  // if this chunk is assigned
            {
                *(pool->alloc_tag[alloci]) = (uint64_t*)((uint64_t)*(pool->alloc_tag[alloci]) + diff);  // dirty: manipulate the pointer directly to shift it accordingly to the reallocated base.
            } else {
                printf("skipped adjust\n");
            }
          }
          for (uint64_t alloci = (pool->chunks >> 1);
                    alloci < (pool->chunks); alloci++)
          {
            pool->alloc_tag[alloci] = NULL;
          }
          return 1;
        }
      return 0;
    }
  }
  printf("did not find a valid entry\n");
  free(pool->base);
  free(pool->alloc_tag);
  exit(0);
}


void
take_ownership(mempool * pool, uint64_t ** dst, uint64_t ** src)
{
    for (uint32_t i = 0; i < pool->chunks; i++)
    {
        if (pool->alloc_tag[i] == src)
        {
            // dst takes over control of src allocated memory
            pool->alloc_tag[i] = dst;
            return;
        }
    }
    printf("did not find src \n");
    exit(0);
}


void
free_pool(mempool * pool, uint64_t * ptr)
{
    if (pool->cur_pop == 0)
    {
        printf("fool, population is currently 0. look for double-free\n");
        exit(0);
    }
    uint64_t pos = (ptr - pool->base) / pool->chunklen;  // position of chunk
    if ((ptr - pool->base) % pool->chunklen != 0)
    {
        printf("failure on free\n");
        free(pool->base);
        exit(0);
    } else {
        pool->alloc_tag[pos] = NULL;
        pool->cur_pop--;
        ptr = NULL;
    }
}
