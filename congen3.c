/* Copyright 2021 Francis Birck Moreira GPL */

#include <malloc.h>
#include <immintrin.h>
#include <stdint.h>
#include "../MC_scrambler/fbm_allocator.h"
#include "../MC_scrambler/avx_func.h"
#include <float.h>
#include "../MC_scrambler/queue.h"
#include <sys/mman.h>

#define MIN_BUF 32
#define XBITS 3
#define CHANNEL_BITS 2
#define BANK_BITS 3
#define COLROW_BITS 7
#define BIT_IGNORE 6
#define TOTAL_BITS 30
#define SOL_STACK_SIZE 14 // 2^14 (16384) positions. WCET We have 11440 combinations (choose 9 out of 16), but bound will eliminate many of them.
#define TEMPSOL_STACK_SIZE 13 // we can create a maximum of half of all combinations

#define RB_HIT 32
#define RB_MISS 128

#define FIX_BANK 143660
#define FIX_XR   231394


typedef struct BBSolBits {
    uint64_t SolBits;
    uint64_t alpha;
    uint64_t BankBits;
    uint64_t RowXorBits;
    uint32_t perm[XBITS];
} BBSolBits;

/* for the future, regresssion tests: do not use this function. changing the solution stack midway iteration has awful and bugg consequences
//=============================================================================
static inline uint64_t 
ForcedSelection(uint64_t Sol_bitcount, uint32_t k, uint64_t * SolBit_stack, uint64_t ** SolSet_stack, uint64_t ** HE, uint64_t i, int64_t j, uint64_t UnionSize)
{
  uint32_t auxi = i;
  while (Sol_bitcount < k)
  {
    SolBit_stack[j] |= (1 << auxi);//adds bit i to Sol
    bmset_union(SolSet_stack[j], HE[auxi], UnionSize);
    Sol_bitcount++;
    auxi++;
  }
  return bmset_popcnt(SolSet_stack[j], UnionSize);
}*/
//=============================================================================

//=============================================================================
//=============================================================================
static inline BBSolBits
BB_iterative(uint64_t ** HE, uint64_t HE_set_size, uint32_t k,
        uint64_t upperbound, uint64_t total_queue_penalty, uint8_t * weights, uint64_t * RowXor_HE_bitmap, mempool * pool,
        uint64_t * SolBit_stack, uint64_t ** SolSet_stack,
        uint64_t * NewSolBit_stack, uint64_t ** NewSolSet_stack)
{
    BBSolBits min;
    min.alpha = upperbound;
    min.SolBits = 0;

    allocpool(pool, &(SolSet_stack[0]));
    bmset_union(SolSet_stack[0], RowXor_HE_bitmap, pool->chunklen);
    SolBit_stack[0] = 0;
    uint64_t Sol_stack_size = 1;  // Fset stack size, we now have 1 element on the stack

// now we have a base F with a single hedge that has xrow's flips, we must find hedge sets of size k that have the minimum bit flips.
    for (uint64_t i = 0; (i < HE_set_size && Sol_stack_size > 0); i++) //2nd stop condition may happen if we eliminate all solutions in the stack due to bounding (i.e., all solutions will be larger than our current best solution, so we preemptively remove them)
    {
        uint64_t NewSol_stack_size = 0;
        for (int64_t j = Sol_stack_size-1; j >= 0; j--)
        {
      uint32_t Sol_bitcount = _mm_popcnt_u64(SolBit_stack[j]);
      uint32_t bits_to_select = k - Sol_bitcount;
      uint32_t remaining_bits = HE_set_size - i;
      uint32_t isForcedSelection = remaining_bits <= bits_to_select;  
            if (isForcedSelection)  // bound: must add every edge to this F, so no point in adding a new F. just add remaining bits to the existent one, calc and quit.
            {
        SolBit_stack[j] |= (1 << i);
        bmset_union(SolSet_stack[j], HE[i], pool->chunklen);
        if (Sol_bitcount + 1 == k)
        {
          uint64_t new_alpha = total_queue_penalty + bmset_calculate_alpha(SolSet_stack[j], weights, pool->chunklen); 
          uint32_t canImprove = (new_alpha < min.alpha);  // branchless technique here, we had awful performance prior to this. TODO changing to <= causes bugs
          min.alpha = (canImprove * new_alpha) + ((!canImprove) * min.alpha);
          min.SolBits = (canImprove * SolBit_stack[j]) + ((!canImprove) * min.SolBits);
          // delete SolSet_stack[j];
          free_pool(pool, SolSet_stack[j]);
          Sol_stack_size--;  // afterall we just deleted someone.
        }
            } else {  // if we have an F where we have options
                allocpool(pool, &(NewSolSet_stack[NewSol_stack_size]));
                bmset_union(NewSolSet_stack[NewSol_stack_size], SolSet_stack[j], pool->chunklen);
                NewSolBit_stack[NewSol_stack_size] = SolBit_stack[j] | (1 << i);
                uint64_t NewSol_bitcount = Sol_bitcount + 1;
                bmset_union(NewSolSet_stack[NewSol_stack_size], HE[i], pool->chunklen);

                uint64_t new_alpha = total_queue_penalty + bmset_calculate_alpha(NewSolSet_stack[NewSol_stack_size], weights, pool->chunklen);
        
        uint32_t OptSol = (new_alpha < min.alpha); //TODO changing to <= causes a segfault
                if (!OptSol) 
        { //bound
                  free_pool(pool, NewSolSet_stack[NewSol_stack_size]);  // no need to calculate solutions above our best solution.
                } else {
          uint32_t SelectedK = (NewSol_bitcount == k);    
          if (SelectedK) //found a better/equal solution with k bits
                    {
        //    printf("new alpha = %lf, prev alpha = %lf\n", new_alpha, min.alpha);
                        min.alpha = new_alpha;
                        min.SolBits = NewSolBit_stack[NewSol_stack_size];
                        free_pool(pool, NewSolSet_stack[NewSol_stack_size]);
                    } else {  // respects upperbound but still does not have k elements.
                        NewSol_stack_size++;  // we effectively added an element
                    }
        }
            }
        }
        // add all new guys to end of stack
        for (uint64_t ni = 0; ni < NewSol_stack_size; ni++)
        {
            SolBit_stack[Sol_stack_size] = NewSolBit_stack[ni];  // add all new guys to end of stack
            SolSet_stack[Sol_stack_size] = NewSolSet_stack[ni];
            // take possession
            take_ownership(pool, &(SolSet_stack[Sol_stack_size]), &(NewSolSet_stack[ni]));
            Sol_stack_size++;
        }
    }
    // cleanup
    for (uint32_t i = 0; i < Sol_stack_size; i++)
        free_pool(pool, SolSet_stack[i]);
    return min;
}
//=============================================================================
//=============================================================================

//=============================================================================
static inline uint64_t
remove_n_bit(uint64_t addr, uint64_t n)
{
    uint64_t upper = addr & 0xFFFFFFFFFFFFFFFE << n;
    uint64_t lower = addr & ((1 << n) - 1);
    return (upper >> 1) | lower;
}
//=============================================================================

//=============================================================================
static inline uint64_t 
gen_buckets(uint64_t ** buckets, uint8_t ** weights, uint32_t * bsizes, uint64_t * addr, uint64_t * timestamps, uint8_t * type_weights, uint64_t aseq_len, uint32_t * masks1, uint32_t * masks2)
{
  uint32_t total_channelbanks = 1 << (BANK_BITS + CHANNEL_BITS);
  uint32_t amaskb[XBITS];
  uint32_t amaskxr[XBITS];
  uint64_t total_queue_penalty = 0;
  for (uint32_t i = 0; i < XBITS; i++)
  {
      amaskb[i] = 1 << masks1[i]; 
      amaskxr[i] = 1 << masks2[i];
  }
  
  queue * Q = (queue*)malloc(sizeof(queue)*total_channelbanks);
  for (uint32_t i = 0; i < total_channelbanks; i++)
  {
      init_queue(&Q[i], MIN_BUF); 
  }

  for (uint64_t i = 0; i < aseq_len; i++)
  {
    uint32_t bank = 0;
    for (uint64_t j = 0; j < XBITS; j++)
    {
        uint32_t bank_and = (addr[i] & amaskb[j]);
        uint32_t rowxor_and = (addr[i] & amaskxr[j]);
        uint32_t bankval = bank_and >> masks1[j];
        uint32_t rowxorval = rowxor_and >> masks2[j];
        uint32_t bankrowxor_val = (bankval ^ rowxorval ? 1 : 0) << j;
        bank += bankrowxor_val;
    }
    for (uint32_t j = 0; j < CHANNEL_BITS; j++)
    {
      uint32_t ch_and = (addr[i] & (1 << j));
      uint32_t ch_val = ch_and >> j;
      bank += ch_val << (j + XBITS); // reaching to 32 banks
    }
    buckets[bank][bsizes[bank]] = addr[i];
    weights[bank][bsizes[bank]] = type_weights[i]; // initialize a weight in case the request gets ignored
    bsizes[bank]++;
    // with bank idx, check queue logic
    //first, dequeue done requests
    while ((!(empty(&Q[bank]))) && Q[bank].buf[0].end_tstamp <= timestamps[i])
    {
      info aux = dequeue(&Q[bank]);
      weights[bank][aux.index] = (uint8_t)((Q[bank].cursize + 1) * aux.type_weight); // update either the general weights or bank weights vector
      total_queue_penalty += Q[bank].cursize;
    }
    //and now enqueue this request
    info aux1;
    aux1.index = bsizes[bank] - 1; //before the increment, as in line 188
    aux1.type_weight = type_weights[i];
    if (empty(&Q[bank]))
    {
      aux1.end_tstamp = timestamps[i] + RB_MISS;
    } else {
      aux1.end_tstamp = Q[bank].buf[Q[bank].cursize - 1].end_tstamp + RB_MISS;
    }
//    aux1.end_tstamp = (is_empty * timestamps[i]) + ( !(is_empty) * Q[bank].buf[Q[bank].cursize - 1].end_tstamp) + RB_MISS;
    enqueue(&Q[bank],aux1);
  }

  //empty queues
  for (uint32_t bank = 0; bank < total_channelbanks; bank++)
  {
    while (!(empty(&Q[bank]))) 
    {
      info aux = dequeue(&Q[bank]);
      weights[bank][aux.index] = (uint8_t)((Q[bank].cursize + 1) * aux.type_weight); // update either the bank weights vector
      total_queue_penalty += Q[bank].cursize;
    }
    total_queue_penalty += MIN_BUF*Q[bank].ignored_reqs; //requests that were ignored due to extra weight
    destroy_queue(&Q[bank]); // Q[b] is now empty, we can free it
  }
  free(Q);
  return total_queue_penalty;
}
  //=============================================================================

//=============================================================================
static inline void
rm_bucket_bits(uint64_t ** buckets, uint64_t bbsize, uint32_t *bsizes, uint32_t * bits, uint64_t bitsize)
{
    for (uint64_t i = 0; i < bbsize; i++)
    {
        for (uint64_t j = 0; j < bsizes[i]; j++)
        {
            uint64_t aux = buckets[i][j];
            for (uint64_t k = 0; k < bitsize; k++)
            {
                aux = remove_n_bit(aux, bits[k]);
            }
            buckets[i][j] = aux >> CHANNEL_BITS; // remove  the channel bits
        }
    }
}
//=============================================================================

//=============================================================================
static void
gen_aseq(uint64_t ** bank_buckets, uint8_t ** weights, uint32_t bbsize, uint32_t * bsizes, uint64_t * aseq, uint8_t * aseq_weights)
{
    uint32_t end = bbsize;
    uint64_t aseqend = 0;
    for (uint32_t i = 0; i < end; i++)
    {
        if (bsizes[i] > 0)
        {
            uint32_t invert = (aseqend != 0) && ((aseq[aseqend] ^ bank_buckets[i][0]) == 0);
            for (uint64_t j = 0; j < bsizes[i]; j++)
            {
            //  aseq[aseqend] = invert ? ~bank_buckets[i][j] : bank_buckets[i][j];
                aseq[aseqend] = (invert*(~bank_buckets[i][j])) + (!(invert)*bank_buckets[i][j]);
                aseq_weights[aseqend] = weights[i][j];
                aseqend++;
            }
        }
    }
    return;
}
//=============================================================================

//=============================================================================
static inline void
gen_edges(uint64_t ** E, uint64_t * aseq, uint64_t aseq_len, uint64_t n, mempool * pool)
{
    for (uint64_t i = 0; i < n; i++)
    {
        allocpool(pool, &(E[i]));
    }
    uint64_t lastaddr = 0;
    for (uint64_t i = 0; i < aseq_len; i++)
    {
        uint64_t toggles = aseq[i] ^ lastaddr;
        for (uint64_t j = 0; j < n; j++)
        {
            bmset_setbit(E[j], i, toggles & 1);
            toggles >>= 1;
        }
        lastaddr = aseq[i];
    }
}
//=============================================================================

//=============================================================================
static inline void
gen_rowstar_edges(uint64_t * rxset, uint64_t ** buckets, uint8_t ** weights, uint64_t nbuckets, uint32_t * bsizes, uint64_t * aseq, uint8_t * aseq_weights, uint64_t aseq_len, uint64_t rowxmask)
{
    gen_aseq(buckets, weights, nbuckets, bsizes, aseq, aseq_weights);
    uint64_t naddr = aseq_len;
    uint64_t lastaddr  = 0;
    for (uint64_t i = 0; i < naddr; i++)
    {
        uint64_t toggles = aseq[i] ^ lastaddr;
        uint64_t isit = toggles & rowxmask ? 1 : 0;
        bmset_setbit(rxset, i, isit);
        lastaddr = aseq[i];
    }
}
//=============================================================================

//=============================================================================
inline void
update_solutions(uint64_t * cur_alpha, BBSolBits aux, uint32_t * curminpop, BBSolBits * minbb, uint64_t * minbb_size, uint32_t ** perms, uint64_t perms_idx)
{
  if ((*cur_alpha) > aux.alpha)  // eliminate previous Sols, this is better
  {
    (*cur_alpha) = aux.alpha;
    printf("changed minbb, alpha is %lu, xrmask is %lu, bankmask is %lu, SolBits is %lu\n", aux.alpha, aux.RowXorBits, aux.BankBits, aux.SolBits);
    // Erase previous "mins", we found a better Sol.
    for (uint32_t mini = 0; mini < (*curminpop); mini++)
      minbb[mini].alpha = 0;
    
    // Set new first minbb
    minbb[0] = aux;
    memcpy(minbb[0].perm, perms[perms_idx], XBITS * sizeof(uint32_t));
    (*curminpop) = 1;
  } else {
    if ((*cur_alpha == aux.alpha) && (aux.SolBits != 0))  // must add new valid Sol
    {
      minbb[*curminpop] = aux;
    //  minbb[curminpop].perm = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
      memcpy(minbb[(*curminpop)].perm, perms[perms_idx], XBITS * sizeof(uint32_t));
      (*curminpop)++;
      if ((*curminpop) >= (*minbb_size))
      {
        (*minbb_size) <<= 0x1ULL;
        minbb = (BBSolBits*)realloc(minbb, (*minbb_size));
        if (minbb == NULL)
        {
          printf("failed to realloc min\n");
          exit(0);
        }
      }
    }
  }
}
//=============================================================================

//=============================================================================
inline void
create_permutations( uint64_t bankmask, uint64_t xrmask, uint32_t * xr_bits, uint32_t * bb_bits, uint32_t ** perms, uint64_t totperms)
{
  // generate permutations of xr bits to combine the masks.
  uint32_t baseb = 0;
  uint32_t basex = 0;
  for (uint32_t bsearch = 0; bsearch < 63 && basex < XBITS; bsearch++)
  {
    uint32_t foundx = (xrmask >> bsearch) & 1;
    xr_bits[basex] = (foundx*bsearch) + (!(foundx)*xr_bits[basex]);
    basex += foundx;
  }
  for (uint32_t bsearch = 0; bsearch < 63 && baseb < XBITS; bsearch++)
  {
    uint32_t foundb = (bankmask >> bsearch) & 1;
    bb_bits[baseb] = (foundb*bsearch) + ((!foundb)*bb_bits[baseb]);
    baseb += foundb;
  }

  for (uint32_t i = 0; i < totperms; i++)
  {
    gen_next_perm(&xr_bits[0], &xr_bits[XBITS]);
    for (uint32_t j = 0; j < XBITS; j++)
      perms[i][j] = xr_bits[j];
  }

}
//=============================================================================

//=============================================================================
static inline void
gen_singlebank_rowstar_edges(uint64_t * rxset, uint64_t * bank_aseq, uint64_t bank_aseq_len, uint64_t rowxmask)
{
  uint64_t lastaddr  = 0;
  for (uint64_t i = 0; i < bank_aseq_len; i++)
  {
    uint64_t toggles = bank_aseq[i] ^ lastaddr;
    uint32_t isit = toggles & rowxmask ? 1 : 0;
    bmset_setbit(rxset, i, isit);
    lastaddr = bank_aseq[i];
  }
}
//=============================================================================

//=============================================================================
void
single_solver(BBSolBits * bankmaps, uint64_t * addresses, uint64_t * timestamps, uint8_t * type_weights, uint64_t aseq_len, uint32_t BankBits, uint32_t rowbits, uint32_t colbits, uint64_t bankmask, uint64_t xrmask, uint32_t xrbitorder[XBITS])
{
  uint32_t total_banks = 1 << (CHANNEL_BITS + BANK_BITS); 
  uint64_t ** buckets = (uint64_t**)malloc(sizeof(uint64_t*)*total_banks);
  uint8_t ** weights = (uint8_t**)malloc(sizeof(uint8_t*)*total_banks);
  uint32_t * bsizes = (uint32_t*)malloc(sizeof(uint32_t)*total_banks);
  uint32_t * bsizes_alloc = (uint32_t*)malloc(sizeof(uint32_t)*total_banks);
  uint64_t HE_set_size = rowbits + colbits - XBITS;
  uint64_t ** HE = (uint64_t**)malloc(sizeof(uint64_t*) * HE_set_size);
  const uint64_t avx_align = 64;
 // const uint64_t base8 = 0x8ULL;
 // const uint64_t and_mod8 = 0x7ULL;
  const uint64_t and_mod64 = 0x3FULL;

 // uint64_t addr_leftover = aseq_len & and_mod8;  // & 7 means % 8.. we want to allocate aligned and multiple to/of 64~B, so we make a check of whether we have a multiple of 8 uint64_t. If we don't, add the leftover.
//  uint64_t addr_roundup = base8 - addr_leftover; // add this to get a multiple of 64
//  uint64_t * aseq = (uint64_t*)aligned_alloc(avx_align, sizeof(uint64_t) * (aseq_len + addr_roundup));

  uint64_t weight_leftover = aseq_len & and_mod64; // bytes leftover. 
  uint64_t weight_roundup = avx_align - weight_leftover; //number of bytes to get to 64
 // uint8_t * aseq_weights = (uint8_t*)aligned_alloc(avx_align, sizeof(uint8_t) * (aseq_len + weight_roundup));
 // memset(aseq_weights,0,(aseq_len + weight_roundup) * sizeof(uint8_t));
  // ---------------------------------------------------------------------
  // bb_iterative mallocs
  uint64_t * SolBit_stack = malloc(sizeof(uint64_t)* (1 << SOL_STACK_SIZE));
  uint64_t ** SolSet_stack = malloc(sizeof(uint64_t*)* (1 << SOL_STACK_SIZE));
  uint64_t * NewSolBit_stack = (uint64_t*)malloc(sizeof(uint64_t) * (1 << TEMPSOL_STACK_SIZE));
  uint64_t ** NewSolSet_stack = (uint64_t**)malloc(sizeof(uint64_t) * (1 << TEMPSOL_STACK_SIZE));
  // ---------------------------------------------------------------------
  // Allocate a memory pool for the thread to avoid malloc/free
  mempool mpool;
  initpool(&mpool, aseq_len, 64);  // predefined estimate, might be reallocated
  for (uint32_t i = 0; i < total_banks; i++)
  {
    memset(bsizes, 0, sizeof(uint32_t)*total_banks);
    buckets[i] = (uint64_t*)malloc(sizeof(uint64_t)* (aseq_len + 1));
    weights[i] = (uint8_t*)aligned_alloc(avx_align, sizeof(uint8_t) * (aseq_len + weight_roundup));
    memset(weights[i], 0, sizeof(uint8_t) * (aseq_len + weight_roundup));
    bsizes_alloc[i] = aseq_len + 1;
  }
  
  uint32_t bb_bits[XBITS];
  uint32_t xr_bits[XBITS];
  uint32_t baseb = 0;
  for (uint32_t bsearch = 0; bsearch < 63 && baseb < XBITS; bsearch++)
  {
    uint32_t foundb = (bankmask >> bsearch) & 1;
    bb_bits[baseb] = (foundb*bsearch) + ((!foundb)*bb_bits[baseb]);
    baseb += foundb;
  }
  uint32_t basex = 0;
  for (uint32_t bsearch = 0; bsearch < 63 && basex < XBITS; bsearch++)
  {
    uint32_t foundx = (xrmask >> bsearch) & 1;
    xr_bits[basex] = (foundx*bsearch) + (!(foundx)*xr_bits[basex]);
    basex += foundx;
  }
  uint32_t totalbrbits = BankBits + XBITS;
  uint32_t * brbits = (uint32_t*)malloc(sizeof(uint32_t)*(totalbrbits));

  uint64_t total_queue_penalty = gen_buckets(buckets, weights, bsizes, addresses, timestamps, type_weights, aseq_len, bb_bits, xrbitorder);

//----------------------------------------------------
// Now per bank structures
//----------------------------------------------------

  uint64_t ** rowx_star = (uint64_t**)malloc(sizeof(uint64_t*)*total_banks);
  // do for each bank an individual rowstar
  for ( uint32_t bsol = 0; bsol < total_banks; bsol++)
  {
    allocpool(&mpool, &rowx_star[bsol]);
    gen_singlebank_rowstar_edges(rowx_star[bsol], buckets[bsol], bsizes[bsol], xrmask);
  }

  for (uint32_t i = 0; i < BankBits; i++) { brbits[i] = bb_bits[i]; }
  for (uint32_t i = BankBits; i < totalbrbits; i++) { brbits[i] = xrbitorder[i - BankBits]; }
      
  qsort(brbits, totalbrbits, sizeof(uint32_t), qs_cmpfunc_desc); // order descending so remove works properly
  rm_bucket_bits(buckets, total_banks, bsizes, brbits, totalbrbits);
  for (uint32_t i = 0; i < total_banks; i++)
  {
    // generate hyperedges for every relevant bit
    gen_edges(HE, buckets[i], bsizes[i], HE_set_size, &mpool);
    uint64_t cur_alpha = UINT64_MAX;
    BBSolBits aux = BB_iterative(HE, HE_set_size, rowbits - XBITS,
                              cur_alpha, total_queue_penalty, weights[i], rowx_star[i], &mpool,
                              SolBit_stack, SolSet_stack, NewSolBit_stack, NewSolSet_stack);
    bankmaps[i].alpha = aux.alpha;
    bankmaps[i].SolBits = aux.SolBits;
    bankmaps[i].BankBits = bankmask;
    bankmaps[i].RowXorBits = xrmask;
    for (uint64_t i = 0; i < HE_set_size; i++)
    {
      free_pool(&mpool, HE[i]);
    }
  }
  
  destroypool(&mpool);  
  for (uint32_t i = 0; i < total_banks; i++)
  {
    free(buckets[i]);
    free(weights[i]);
  }
  free(SolBit_stack);
  free(SolSet_stack); 
  free(NewSolBit_stack);
  free(NewSolSet_stack);
  free(brbits);
  free(buckets); 
  free(weights);
  free(bsizes);
  free(bsizes_alloc); 
  free(HE); 
}
//=============================================================================
//=============================================================================
//BBSolBits *
BBSolBits
solver(uint64_t * addresses, uint64_t * timestamps, uint8_t * type_weights, uint64_t aseq_len, uint32_t BankBits, uint32_t rowbits, uint32_t colbits, uint64_t * total_Sols)
{
    uint32_t n = BankBits + rowbits + colbits;
    uint64_t total_bmask = 0;
    uint64_t ch_mask = 0;
    for (uint32_t i = 0; i < CHANNEL_BITS; i++) { ch_mask += 1 << i;}//fixed channel bits
    uint64_t * bank_masks = do_masked_combinations(n, BankBits, &total_bmask, ch_mask);
//    uint64_t minbb_size = MIN_BUF;
//    BBSolBits * minbb = (BBSolBits*)malloc(sizeof(BBSolBits)*minbb_size);
  //  memset(minbb, 0, sizeof(BBSolBits)*minbb_size);
    BBSolBits minbb;


    uint32_t curminpop = 1;
    uint64_t cur_alpha = UINT64_MAX;
    uint64_t totperms = 1;
    for (uint32_t i = 1; i <= XBITS; i++) { totperms *= i; }

    #pragma omp parallel for
    for (uint64_t bmask_idx = 0; bmask_idx < total_bmask; bmask_idx++)
    {
        uint64_t bankmask = bank_masks[bmask_idx];
        uint64_t total_rmask = 0;
        // allocations to be used (inside loop to make sure threads do not conflict)
        // --------------------------------------------------------------------
        uint64_t * xrow_masks = do_masked_combinations(n - BankBits, XBITS, &total_rmask, bankmask | ch_mask); //skip both bank and ch masks
        uint32_t * xr_bits = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
        memset(xr_bits, 0, sizeof(uint32_t)*XBITS);
        uint32_t * bb_bits = (uint32_t*)malloc(sizeof(uint32_t)*BankBits);
        memset(bb_bits, 0, sizeof(uint32_t)*BankBits);
        uint32_t total_banks = 1 << (BANK_BITS + CHANNEL_BITS);
        uint64_t ** buckets = (uint64_t**)malloc(sizeof(uint64_t*)*total_banks);
        uint8_t ** weights = (uint8_t**)malloc(sizeof(uint8_t*)*total_banks);
        uint32_t * bsizes = (uint32_t*)malloc(sizeof(uint32_t)*total_banks);
        uint32_t * bsizes_alloc = (uint32_t*)malloc(sizeof(uint32_t)*total_banks);
        uint64_t HE_set_size = rowbits + colbits - XBITS;
        uint64_t ** HE = (uint64_t**)malloc(sizeof(uint64_t*) * HE_set_size);
        
        
        const uint64_t avx_align = 64;
        const uint64_t base8 = 0x8ULL;
        const uint64_t and_mod8 = 0x7ULL;
        const uint64_t and_mod64 = 0x3FULL;

        uint64_t addr_leftover = aseq_len & and_mod8;  // & 7 means % 8.. we want to allocate aligned and multiple to/of 64~B, so we make a check of whether we have a multiple of 8 uint64_t. If we don't, add the leftover.
        uint64_t addr_roundup = base8 - addr_leftover; // add this to get a multiple of 64
        uint64_t * aseq = (uint64_t*)aligned_alloc(avx_align, sizeof(uint64_t) * (aseq_len + addr_roundup));
      
        uint64_t weight_leftover = aseq_len & and_mod64; // bytes leftover. 
        uint64_t weight_roundup = avx_align - weight_leftover; //number of bytes to get to 64
        uint8_t * aseq_weights = (uint8_t*)aligned_alloc(avx_align, sizeof(uint8_t) * (aseq_len + weight_roundup));
        memset(aseq_weights,0,(aseq_len + weight_roundup) * sizeof(uint8_t));
        uint32_t ** perms = (uint32_t**)malloc(sizeof(uint32_t*)*totperms);
        for (uint32_t i = 0; i < totperms; i++)
        {
                perms[i] = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
        }
        uint32_t totalbrbits = BankBits + XBITS;
        uint32_t * brbits = (uint32_t*)malloc(sizeof(uint32_t)*(totalbrbits));
        // ---------------------------------------------------------------------
        // bb_iterative mallocs
        uint64_t * SolBit_stack = malloc(sizeof(uint64_t)* (1 << SOL_STACK_SIZE));
        uint64_t ** SolSet_stack = malloc(sizeof(uint64_t*)*(1 << SOL_STACK_SIZE));
        uint64_t * NewSolBit_stack = (uint64_t*)malloc(sizeof(uint64_t) * (1 << TEMPSOL_STACK_SIZE));
        uint64_t ** NewSolSet_stack = (uint64_t**)malloc(sizeof(uint64_t) * (1 << TEMPSOL_STACK_SIZE));
        // ---------------------------------------------------------------------
        // Allocate a memory pool for the thread to avoid malloc/free
        mempool mpool;
        initpool(&mpool, aseq_len, 64);  // predefined estimate, might be reallocated
        // got the xrows as well.
        for (uint64_t xrmask_idx = 0; xrmask_idx < total_rmask; xrmask_idx++)
        {
            uint64_t xrmask = xrow_masks[xrmask_idx];
            create_permutations(bankmask, xrmask, xr_bits, bb_bits, perms, totperms);

            for (uint32_t perms_idx = 0; perms_idx < totperms; perms_idx++)
            {
                memset(bsizes, 0, sizeof(uint32_t)*total_banks);
                for (uint32_t i = 0; i < total_banks; i++)
                {
                    buckets[i] = (uint64_t*)malloc(sizeof(uint64_t)* (aseq_len + 1));
                    weights[i] = (uint8_t*)aligned_alloc(avx_align, sizeof(uint8_t) * (aseq_len + weight_roundup));
                    memset(weights[i], 0, sizeof(uint8_t) * (aseq_len + weight_roundup));
                    bsizes_alloc[i] = aseq_len + 1;
                }
                uint64_t total_queue_penalty = gen_buckets(buckets, weights, bsizes, addresses, timestamps, type_weights, aseq_len, bb_bits, perms[perms_idx]);
                    
                uint64_t * rowx_star;
                allocpool(&mpool, &rowx_star);
                gen_rowstar_edges(rowx_star, buckets, weights, total_banks, bsizes, aseq, aseq_weights, aseq_len, xrmask);
                for (uint32_t i = 0; i < BankBits; i++) { brbits[i] = bb_bits[i]; }
                for (uint32_t i = BankBits; i < totalbrbits; i++) { brbits[i] = perms[perms_idx][i - BankBits]; }
                    
                qsort(brbits, totalbrbits, sizeof(uint32_t), qs_cmpfunc_desc); // order descending so remove works properly
                rm_bucket_bits(buckets, total_banks, bsizes, brbits, totalbrbits);
                // generate address sequence for the bank xor row bits selected in order defined by permidx, and reorder weights accordingly
                gen_aseq(buckets, weights, total_banks, bsizes, aseq, aseq_weights);

                // generate hyperedges for every relevant bit
                gen_edges(HE, aseq, aseq_len, HE_set_size, &mpool);

                BBSolBits aux = BB_iterative(HE, HE_set_size, rowbits - XBITS,
                                            cur_alpha, total_queue_penalty, aseq_weights, rowx_star, &mpool,
                                            SolBit_stack, SolSet_stack, NewSolBit_stack, NewSolSet_stack);
                aux.BankBits = bankmask;
                aux.RowXorBits = xrmask;
                #pragma omp critical
                {
                    if (aux.alpha < cur_alpha)
                    {
                      cur_alpha = aux.alpha;
                      minbb.BankBits = bankmask;
                      minbb.RowXorBits = xrmask;
                      minbb.SolBits = aux.SolBits;
                      minbb.alpha = aux.alpha;
                      memcpy(minbb.perm, perms[perms_idx], XBITS * sizeof(uint32_t));
                      printf("changed minbb, alpha is %lu, xrmask is %lu, bankmask is %lu, SolBits is %lu\n", aux.alpha, aux.RowXorBits, aux.BankBits, aux.SolBits);
                    }
                    //update_solutions(&cur_alpha, aux, &curminpop, minbb, &minbb_size, perms,perms_idx);
                }
                for (uint64_t i = 0; i < HE_set_size; i++)
                {
                    free_pool(&mpool, HE[i]);
                }
                free_pool(&mpool, rowx_star);
                for (uint32_t bank = 0; bank < total_banks; bank++)
                {
                    free(buckets[bank]);
                    free(weights[bank]);
                }
            }
        }
        // FREE EVERYONE!
        destroypool(&mpool);
        free(xrow_masks);
        free(xr_bits);
        free(bb_bits);
        free(buckets);
        free(weights);
        free(bsizes);
        free(bsizes_alloc);
        free(HE);
        free(aseq);
        free(aseq_weights);

        for (uint32_t i = 0; i < totperms; i++) { free(perms[i]); };
        free(perms);
        free(brbits);
        free(SolSet_stack);
        free(SolBit_stack);
        free(NewSolBit_stack);
        free(NewSolSet_stack);
    }
    free(bank_masks);
    *total_Sols = curminpop;
    return minbb;
}
//=============================================================================
//=============================================================================


//=============================================================================
uint64_t
read_input_addresses(char * inputfile, uint64_t * A, uint64_t * tstamps, uint8_t * type_weights, uint64_t expected_lines)
{
  uint64_t totlines = 0;
  FILE * f = fopen(inputfile, "r");
  uint64_t * B = (uint64_t *)malloc(sizeof(uint64_t) * (expected_lines+10));
  uint64_t newaddr = 0;
  uint64_t cycle = 0;
  char * dummy = (char*)malloc(sizeof(char)*3);
  uint64_t bufferlines = 0;
  uint64_t k = MIN_BUF;  // predefined buffer size, aka high-water mark for memory controller.
  uint32_t prev = 0;  // 0 = read, 1 = write
	const uint8_t pref_weight = 1;
	const uint8_t read_inst_weight = 2;
	const uint8_t wb_weight = 0;
  while (fscanf(f, "%s %lu %lu\n", dummy, &newaddr, &cycle) != EOF)
  {
    if (strcmp(dummy, "WB") != 0)  // not write.
    {
      uint32_t found = 0;
      for (uint32_t j = 0; j < bufferlines; j++)
      {
        if (newaddr == B[j])
        {
          found = 1;
          break;
        }
      }
      if (found)
      {
        for (uint32_t j = 0; j < bufferlines; j++)
        {
          A[totlines] = B[j] >> BIT_IGNORE;
          tstamps[totlines] = cycle;
					type_weights[totlines] = wb_weight;
          totlines++;
        }
        bufferlines = 0;
      }
      A[totlines] = newaddr >> BIT_IGNORE;
      tstamps[totlines] = cycle;
			uint8_t isRI = ( strcmp(dummy,"P") != 0);
			type_weights[totlines] = pref_weight + (isRI*read_inst_weight); //if prefetch, assign pref weight if read or inst, add weight.
      totlines++;
      prev = 0;
    } else {
      if (prev)  // previous was a WRITE (W -> W)
      {
        A[totlines] = newaddr >> BIT_IGNORE;
        tstamps[totlines] = cycle;
				type_weights[totlines] = wb_weight;
        totlines++;
      } else {  // previous was a READ (R -> W)
        if (bufferlines < k)  // add to buffer so we can continue servicing R
        {
          B[bufferlines] = newaddr;
          bufferlines++;
        } else {  // buffer is full, we *have* to drain the writes!
          for (uint32_t j = 0; j < k; j++)
          {
            A[totlines] = B[j] >> BIT_IGNORE;
            tstamps[totlines] = cycle; //draining at this cycle instead of B_tstamp, which represents when the write entered queue. should we count write q delay?
						type_weights[totlines] = wb_weight;
            totlines++;
          }
          bufferlines = 0;
          A[totlines] = newaddr >> BIT_IGNORE;
          tstamps[totlines] = cycle;
					type_weights[totlines] = wb_weight;
          totlines++;
          prev = 1;
        }
      }
    }
  }
  for (uint32_t j = 0; j < bufferlines; j++)
  {
    A[totlines] = B[j] >> BIT_IGNORE;
    tstamps[totlines] = cycle; //whenever we extract from B, we are write draining, so we should always place cycle, no?
    type_weights[totlines] = wb_weight;
    totlines++;
  }
  bufferlines = 0;
  free(B);
  free(dummy);
  fclose(f);
  return totlines;
}
//=============================================================================

//=============================================================================
void
print_solution(BBSolBits * bbmin, uint64_t total_Sols)
{
  for (uint64_t i = 0; i < total_Sols; i++)
  {
    printf("bbmin.alpha %lu\n", bbmin[i].alpha);
    printf("got to end, bbmin SolBits = %lu\n", bbmin[i].SolBits);
    for (uint64_t k = 0; k < 64; k++)
    {
      if (1 & (bbmin[i].SolBits >> k))
        printf("%lu ", k);
    }
    printf("\n");
    printf(" bank mask was %lu\n", bbmin[i].BankBits);
    printf(" xor rows msk was %lu\n", bbmin[i].RowXorBits);
    printf("perm order\n");
    for (uint32_t k = 0; k < XBITS; k++)
    {
      printf("%u ",bbmin[i].perm[k]);
    }
    printf("\n");
  }
}
//=============================================================================

void
print_solution_single(BBSolBits bbmin)
{
    printf("bbmin.alpha %lu\n", bbmin.alpha);
    printf("got to end, bbmin SolBits = %lu\n", bbmin.SolBits);
    for (uint64_t k = 0; k < 64; k++)
    {
      if (1 & (bbmin.SolBits >> k))
        printf("%lu ", k);
    }
    printf("\n");
    printf(" bank mask was %lu\n", bbmin.BankBits);
    printf(" xor rows msk was %lu\n", bbmin.RowXorBits);
    printf("perm order\n");
    for (uint32_t k = 0; k < XBITS; k++)
    {
      printf("%u ",bbmin.perm[k]);
    }
    printf("\n");
}
//=============================================================================
//=============================================================================
int 
main(int argc, char ** argv)
{
  if (argc != 3)
  {
    printf("wrong number of args: %d\n", argc);
    for (int32_t i = 0; i < argc; i++)
      printf("got %s\n", argv[i]);
    exit(0);
  }

  char * inputfile = argv[1];
  uint64_t expected_lines = strtoul(argv[2], NULL, 10);
  //uint64_t * A = (uint64_t *)malloc(sizeof(uint64_t) * (expected_lines+10));
  uint64_t * A = (uint64_t*) mmap(0, sizeof(uint64_t) * (expected_lines+10), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  //uint64_t * timestamps = (uint64_t *)malloc(sizeof(uint64_t) * (expected_lines+10));
  uint64_t * timestamps = (uint64_t*) mmap(0, sizeof(uint64_t) * (expected_lines+10), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS| MAP_HUGETLB, -1, 0);
//	uint8_t * type_weights = (uint8_t *)malloc(sizeof(uint8_t) * (expected_lines+10));
  uint8_t * type_weights = (uint8_t*) mmap(0, sizeof(uint8_t) * (expected_lines+10), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS| MAP_HUGETLB, -1, 0);
  uint64_t totlines = read_input_addresses(inputfile, A, timestamps, type_weights, expected_lines);
  uint64_t total_Sols = 0;
  
  //uint32_t stub_xrbits[XBITS] = {0, 1, 2};
  //BBSolBits * bbmin = (BBSolBits*)malloc(sizeof(BBSolBits)*(1 << (BANK_BITS + CHANNEL_BITS)));
  //single_solver(bbmin, A, timestamps, type_weights, totlines, BANK_BITS, TOTAL_BITS - (BANK_BITS + CHANNEL_BITS + COLROW_BITS + BIT_IGNORE), COLROW_BITS, FIX_BANK, FIX_XR,stub_xrbits);  
  
  
  //----------------------------------------
  
  //BBSolBits * bbmin = solver(A, timestamps, type_weights, totlines, BANK_BITS, TOTAL_BITS - (BANK_BITS + CHANNEL_BITS + COLROW_BITS + BIT_IGNORE), COLROW_BITS, &total_Sols);
  BBSolBits bbmin = solver(A, timestamps, type_weights, totlines, BANK_BITS, TOTAL_BITS - (BANK_BITS + CHANNEL_BITS + COLROW_BITS + BIT_IGNORE), COLROW_BITS, &total_Sols);
//  print_solution(bbmin, total_Sols);
  print_solution_single(bbmin);
  
//

//  free(bbmin);
//  free(A);
  munmap(A, sizeof(uint64_t) * (expected_lines + 10));
//  free(timestamps);
  munmap(timestamps, sizeof(uint64_t) * (expected_lines + 10));
//  free(type_weights);
  munmap(type_weights, sizeof(uint8_t) * (expected_lines + 10));
  return 0;
}
//=============================================================================
//=============================================================================
