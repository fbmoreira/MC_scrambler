/* Copyright 2021 Francis Birck Moreira GPL */

#include<immintrin.h>
#ifndef _AVX_FUNC_H_
    #define _AVX_FUNC_H_
#endif  // _AVX_FUNC_H

static inline int
qs_cmpfunc_asc(const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}

static inline int
qs_cmpfunc_desc(const void * a, const void * b)
{
    return ( *(int*)b - *(int*)a );
}

static inline uint64_t
bmset_popcnt(uint64_t * a, uint64_t len)
{  // 2nd function taking longest on BB_iterative
        uint64_t acc1 = 0;
        for (uint64_t i = 0; i < len; i++)
        {
            acc1+= __builtin_popcountll(a[i]);
        }
        return acc1;
}
static inline void
bmset_union(uint64_t * a, uint64_t * b, uint64_t len)
{
        // chunklen contains #uint64_t, so every 8 we get one m512.
		//printf("received len %lu\n", len);
        uint64_t m512_len = len >> 3;
        for (uint64_t i = 0; i < m512_len; i+=8)  // len is chunklen in uint64_t. so each len represents 8B. we need to cut len by 8 then so avx512 is properly represented.
        {
            __m512i ra = _mm512_load_epi64(&a[i]);
            __m512i rb = _mm512_load_epi64(&b[i]);
            __m512i res1 = _mm512_or_epi64(ra, rb);
            _mm512_store_epi64(&a[i], res1);  // perf accuses this of taking most time.
        }
}

static inline void
bmset_setbit(uint64_t * bitmap, uint64_t n, uint32_t val)
{
        uint64_t pos = n >> 0X6ULL;  // which bitmap pos?
        uint64_t bit = n - (pos << 0x6ULL);
//		printf("trying to set bit %lu in pos %lu\n", bit, pos);
        bitmap[pos] |= val << bit;
}

uint64_t *
do_masked_combinations(uint64_t n, uint64_t k, uint64_t * totalbmask, uint64_t mask)
{
    if (k > n)
        return NULL;
    if ( (k * 2) > n )
        k = n - k;
    if ( k == 0)
        return NULL;
    uint64_t total = n;
    for (uint64_t i = 2; i <= k; i++)
    {
        total *= (n-i+1);
        total /= i;
    }
    uint64_t * combptr = (uint64_t*)malloc(sizeof(uint64_t)*total);
    combptr[0] = 0;
    for (uint64_t i = 0; i < k; i++)
    {
        combptr[0] |= 1 << i;
    }
    uint64_t prev = combptr[0];  // creating a base case.
    uint32_t match = (prev & mask);
    uint64_t i = !match;  // is the base a match to the passed mask?
    // if it is, we cannot count it as a valid permutation due to bit conflict,
    //  so we start adding to 0.
    while (i < total)
    {
        uint64_t t = prev | (prev -1);
        uint64_t x = (t+1) | (((~t & -(~t))-1) >> (__builtin_ctzll(prev)+1));
        prev = x;
        uint32_t match = (x & mask);
        combptr[i] = match ? combptr[i] : x;
        i += match ? 0 : 1;  // do we have a bit conflict with bank mask? if we do, keep index the same so we can replace the conflicting address, otherwise we succesfully added a mask and we progress the index.
    }
    *totalbmask = total;
    return combptr;
}

uint32_t
gen_next_perm(uint32_t * begin, uint32_t * end)
{
    if (begin == end)
        return 0;

    uint32_t * i = begin;
    ++i;
    if (i == end)
    {
        return 0;
    }
    i = end;
    --i;

    while (1)
    {
        uint32_t * j = i;
        --i;

        if (*i < *j)
        {
            uint32_t * k = end;
            while (!(*i < *(--k)))
                /* pass */ {};

            // iter_swap(i, k); ::
            uint32_t aux = *k;
            *k = *i;
            *i = aux;


            // reverse(j, end); ::
            while ((j != end) && (j != --end))
            {
                // iter_swap(j, end)
                aux = *end;
                *end = *j;
                *j = aux;
                j++;
            }
            return 1;
        }

        if (i == begin)
        {
            // reverse(begin, end);
            while ((begin != end) && (begin != --end))
            {
                // iter_swap(j, end)
                uint32_t aux = *end;
                *end = *begin;
                *begin = aux;
                begin++;
            }
            //------------------------
            return 0;
        }
    }
}

