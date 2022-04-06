/* Copyright 2021 Francis Birck Moreira GPL */

#include<immintrin.h>
#include<stdint.h>
#include<string.h>
#include<stdio.h>
#ifndef _AVX_FUNC_H_
    #define _AVX_FUNC_H_
#endif  // _AVX_FUNC_H

static inline int
qs_cmpfunc_asc(const void *a, const void *b)
{
    return ( *(int*)a - *(int*)b );
}

static inline int
qs_cmpfunc_desc(const void *a, const void *b)
{
    return ( *(int*)b - *(int*)a );
}

static inline uint64_t
bmset_popcnt(uint64_t *a, uint64_t len)
{  // 2nd function taking longest on BB_iterative
        uint64_t acc1 = 0;
        for (uint64_t i = 0; i < len; i++)
        {
            acc1+= __builtin_popcountll(a[i]);
        }
        return acc1;
}

static inline void
bmset_union(uint64_t *a, uint64_t *b, uint64_t len)
{
    // chunklen contains #uint64_t, so every 8 we get one m512.
    uint32_t m512_len = len >> 3;
    for (uint32_t i = 0; i < m512_len; i+=8)  // len is chunklen in uint64_t. so each len represents 8B. we need to cut len by 8 then so avx512 is properly represented.
    {
        __m512i ra = _mm512_load_epi64(&a[i]);
        __m512i rb = _mm512_load_epi64(&b[i]);
        __m512i res1 = _mm512_or_epi64(ra, rb);
        _mm512_store_epi64(&a[i], res1);  // perf accuses this of taking most time.
    }
}

static inline void
bmset_setbit(uint64_t *bitmap, uint64_t n, uint64_t val)
{
    uint64_t pos = n >> 0X6ULL;  // which bitmap pos?
    uint64_t bit = n - (pos << 0x6ULL);
    bitmap[pos] |= val << bit;
}

static inline uint64_t
bmset_calculate_alpha(uint64_t *bitmap, uint8_t *weights, uint64_t len)
{
	__m512i acc1 = _mm512_setzero_epi32();
	__m512i acc2 = _mm512_setzero_epi32();
	__m512i acc3 = _mm512_setzero_epi32();
	__m512i acc4 = _mm512_setzero_epi32();
	__m128i zero = _mm_setzero_si128();
  uint16_t * bmp = (uint16_t*) bitmap;
	uint32_t elems = len << 6; //every position of pool has 8B, so we have m512len positions of 64B
	for (uint32_t i  = 0; i < elems; i+=64)
	{
      __mmask16 mask1 = _load_mask16(&(bmp[i >> 4]));
      __m128i l1 = _mm_mask_loadu_epi8(zero, mask1, &(weights[i]));
      __m512i add_op1 = _mm512_cvtepi8_epi32(l1); // should convert 16 uint8_t to 16 uint32_t
      acc1 = _mm512_add_epi32(acc1,add_op1);
      
      __mmask16 mask2 = _load_mask16(&(bmp[(i >> 4) + 1]));
      __m128i l2 = _mm_mask_loadu_epi8(zero, mask2, &(weights[i + 16]));
      __m512i add_op2 = _mm512_cvtepi8_epi32(l2); // should convert 16 uint8_t to 16 uint32_t
      acc2 = _mm512_add_epi32(acc2,add_op2);
      
      __mmask16 mask3 = _load_mask16(&(bmp[(i >> 4) + 2]));
      __m128i l3 = _mm_mask_loadu_epi8(zero, mask3, &(weights[i + 32]));
      __m512i add_op3 = _mm512_cvtepi8_epi32(l3); // should convert 16 uint8_t to 16 uint32_t
      acc3 = _mm512_add_epi32(acc3,add_op3);
      
      __mmask16 mask4 = _load_mask16(&(bmp[(i >> 4) + 3]));
      __m128i l4 = _mm_mask_loadu_epi8(zero, mask4, &(weights[i + 48]));
      __m512i add_op4 = _mm512_cvtepi8_epi32(l4); // should convert 16 uint8_t to 16 uint32_t
      acc4 = _mm512_add_epi32(acc4,add_op4);
	}
  acc1 = _mm512_add_epi32(acc1,acc2);
  acc3 = _mm512_add_epi32(acc3,acc4);
  acc1 = _mm512_add_epi32(acc1,acc3);
	return ((uint64_t)_mm512_reduce_add_epi32(acc1));
}

/*{
	__m512i acc = _mm512_setzero_epi32();
	__m128i zero = _mm_setzero_si128();
  uint16_t * bmp = (uint16_t*) bitmap;
	uint32_t elems = len << 6; //every position of pool has 8B, so we have m512len positions of 64B
	for (uint32_t i  = 0; i < elems; i+=16)
	{
      __mmask16 mask = _load_mask16(&(bmp[i >> 4]));
      __m128i l = _mm_mask_loadu_epi8(zero, mask, &(weights[i]));
      __m512i add_op = _mm512_cvtepi8_epi32(l); // should convert 16 uint8_t to 16 uint32_t
      acc = _mm512_add_epi32(acc,add_op);
	}

	return ((uint64_t)_mm512_reduce_add_epi32(acc));
}*/

uint64_t *
do_masked_combinations(uint64_t n, uint64_t k, uint64_t *totalbmask, uint64_t mask)
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

    uint64_t *combptr = (uint64_t*)malloc(sizeof(uint64_t)*total);
    memset(combptr, 0, sizeof(uint64_t)*total);
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
        uint64_t t = prev | (prev - 1);
        uint64_t x = (t+1) | (((~t & -(~t)) - 1) >> (__builtin_ctzll(prev) + 1));
        prev = x;
        uint32_t match = (x & mask);
        combptr[i] = match ? combptr[i] : x;
        i += match ? 0 : 1;  // do we have a bit conflict with bank mask? if we do, keep index the same so we can replace the conflicting address, otherwise we succesfully added a mask and we progress the index.
    }

    *totalbmask = total;
    return combptr;
}

uint32_t
gen_next_perm(uint32_t *begin, uint32_t *end)
{
    if (begin == end)
        return 0;

    uint32_t *i = begin;
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
            uint32_t *k = end;
            while (!(*i < *(--k))) /* pass */ {};

            // iter_swap(i, k);
            uint32_t aux = *k;
            *k = *i;
            *i = aux;

            // reverse(j, end);
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

            return 0;
        }
    }
}
