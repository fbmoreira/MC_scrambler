/* Copyright 2021 Francis Birck Moreira GPL */

#include <malloc.h>
#include <immintrin.h>
#include <stdint.h>
#include "../MC_scrambler/pool_allocator.h"
#include "../MC_scrambler/avx_func.h"

#define MIN_BUF 32
#define XBITS 3
#define BANK_BITS 3
#define COLROW_BITS 7
#define BIT_IGNORE 8
#define TOTAL_BITS 30
#define SOL_STACK_SIZE 14 // 2^20 positions
#define TEMPSOL_STACK_SIZE 10 // 2^10 positions

typedef struct BBSolBits {
    uint64_t SolBits;
    uint64_t alpha;
    uint64_t BankBits;
    uint64_t RowXorBits;
	uint32_t perm[XBITS];
} BBSolBits;


/* for the future, regresssion tests: do not use this function. changing the solution stack midway iteration has awful and bugg consequences
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

static inline BBSolBits
BB_iterative(uint64_t ** HE, uint64_t HE_set_size, uint32_t k,
        uint64_t upperbound, uint64_t * RowXor_HE_bitmap, mempool * pool,
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
		//				uint64_t new_alpha = ForcedSelection(Sol_bitcount, k, SolBit_stack, SolSet_stack, HE, i, j, pool->chunklen);
				SolBit_stack[j] |= (1 << i);
				bmset_union(SolSet_stack[j], HE[i], pool->chunklen);
				if (Sol_bitcount + 1 == k)
				{
					uint64_t new_alpha = bmset_popcnt(SolSet_stack[j], pool->chunklen);
					uint32_t canImprove = (new_alpha < min.alpha);  // branchless technique here, we had awful performance prior to this. TODO changing to <= causes bugs
					min.alpha = (canImprove * new_alpha) + ((!canImprove) * min.alpha);
					min.SolBits = (canImprove * SolBit_stack[j]) + ((!canImprove) * min.SolBits);
					// delete SolSet_stack[j];
					free_pool(pool, SolSet_stack[j]);
					Sol_stack_size--;  // afterall we just deleted someone.
				}
            } else {  // if we have an F where we have options
                uint32_t reloc =  allocpool(pool, &(NewSolSet_stack[NewSol_stack_size]));
                bmset_union(NewSolSet_stack[NewSol_stack_size], SolSet_stack[j], pool->chunklen);
                NewSolBit_stack[NewSol_stack_size] = SolBit_stack[j] | (1 << i);
				uint64_t NewSol_bitcount = Sol_bitcount + 1;
                bmset_union(NewSolSet_stack[NewSol_stack_size], HE[i], pool->chunklen);
                uint64_t new_alpha = bmset_popcnt(NewSolSet_stack[NewSol_stack_size], pool->chunklen);
				uint32_t OptSol = (new_alpha < min.alpha); //TODO changing to <= causes a segfault
                if (!OptSol) 
				{ //bound
                	free_pool(pool, NewSolSet_stack[NewSol_stack_size]);  // no need to calculate solutions above our best solution.
                } else {
					uint32_t SelectedK = (NewSol_bitcount == k);    
					if (SelectedK) //found a better/equal solution with k bits
                    {
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

static inline uint64_t
remove_n_bit(uint64_t addr, uint64_t n)
{
    uint64_t upper = addr & 0xFFFFFFFFFFFFFFFE << n;
    uint64_t lower = addr & ((1 << n) - 1);
    return (upper >> 1) | lower;
}

static inline void
gen_buckets(uint64_t ** buckets, uint32_t * bsizes, uint64_t * addr, uint64_t aseq_len, uint32_t * masks1, uint32_t * masks2, uint32_t num_bits)
{
	uint32_t amaskb[XBITS];
	uint32_t amaskxr[XBITS];
	for (uint32_t i = 0; i < XBITS; i++)
	{
		amaskb[i] = 1 << masks1[i];
		amaskxr[i] = 1 << masks2[i];
	}

    for (uint64_t i = 0; i < aseq_len; i++)
    {
        uint64_t buck = 0;
        for (uint64_t j = 0; j < num_bits; j++)
        {
        	uint32_t m1 = (addr[i] & amaskb[j]) >> masks1[j];
			uint32_t m2 = (addr[i] & amaskxr[j]) >> masks2[j];
		    uint32_t val = (m1 ^ m2 ? 1 : 0) << j;
			buck += val;
        }
        buckets[buck][bsizes[buck]] = addr[i];
        bsizes[buck]++;
    }
}

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
            buckets[i][j] = aux;
        }
    }
}

static inline void
gen_aseq(uint64_t ** bank_buckets, uint64_t bbsize, uint32_t * bsizes, uint64_t * aseq)
{
    uint64_t end = bbsize;
    uint64_t aseqend = 0;
    for (uint64_t i = 0; i < end; i++)
    {
        if (bsizes[i] > 0)
        {
            uint32_t invert = (aseqend != 0) && ((aseq[aseqend] ^ bank_buckets[i][0]) == 0);
            for (uint64_t j = 0; j < bsizes[i]; j++)
            {
            //  aseq[aseqend] = invert ? ~bank_buckets[i][j] : bank_buckets[i][j];
                aseq[aseqend] = (invert*(~bank_buckets[i][j])) + (!(invert)*bank_buckets[i][j]);
                aseqend++;
            }
        }
    }
    return;
}

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

static inline void
gen_rowstar_edges(uint64_t * rxset, uint64_t ** buckets, uint64_t nbuckets, uint32_t * bsizes, uint64_t * aseq, uint64_t aseq_len, uint64_t rowxmask)
{
    gen_aseq(buckets, nbuckets, bsizes, aseq);
    uint64_t naddr = aseq_len;
    uint64_t lastaddr  = 0;
    for (uint64_t i = 0; i < naddr; i++)
    {
        uint64_t toggles = aseq[i] ^ lastaddr;
		uint32_t isit = toggles & rowxmask ? 1 : 0;
		bmset_setbit(rxset, i, isit);
        lastaddr = aseq[i];
    }
}

inline void
update_solutions(uint64_t * cur_alpha, BBSolBits aux, uint32_t * curminpop, BBSolBits * minbb, uint64_t * minbb_size, uint32_t ** perms, uint64_t perms_idx)
{
	if ((*cur_alpha) > aux.alpha)  // eliminate previous Sols, this is better
	{
		printf("changed minbb, alpha is %lu, xrmask is %lu, bankmask is %lu, SolBits is %lu\n", aux.alpha, aux.RowXorBits, aux.BankBits, aux.SolBits);
		// Erase previous "mins", we found a better Sol.
		for (uint32_t mini = 0; mini < (*curminpop); mini++)
			minbb[mini].alpha = 0;
		
		// Set new first minbb
		minbb[0] = aux;
		memcpy(minbb[0].perm, perms[perms_idx], XBITS * sizeof(uint32_t));
		(*curminpop) = 1;
		(*cur_alpha) = aux.alpha;
	} else {
		if ((*cur_alpha == aux.alpha) && (aux.SolBits != 0))  // must add new valid Sol
		{
			printf("adding Sol, current number %u, current minbb_size %lu\n", (*curminpop), (*minbb_size));
			minbb[*curminpop] = aux;
		//	minbb[curminpop].perm = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
			memcpy(minbb[(*curminpop)].perm, perms[perms_idx], XBITS * sizeof(uint32_t));
			(*curminpop)++;
			if ((*curminpop) >= (*minbb_size))
			{
				printf("reallocating previous minbb_size = %lu\n", *minbb_size);
				(*minbb_size) <<= 0x1ULL;
				printf("reallocating new minbb_size = %lu\n", *minbb_size);
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

BBSolBits *
solver(uint64_t * addresses, uint64_t aseq_len, uint32_t BankBits, uint32_t rowbits, uint32_t colbits, uint64_t * total_Sols)
{
    uint32_t n = BankBits + rowbits + colbits;
    uint64_t total_bmask = 0;
    uint64_t * bank_masks = do_masked_combinations(n, BankBits, &total_bmask, 0);
	uint64_t minbb_size = MIN_BUF;
    BBSolBits * minbb = (BBSolBits*)malloc(sizeof(BBSolBits)*minbb_size);
	uint32_t curminpop = 1;
    uint64_t cur_alpha = aseq_len;
	uint64_t totperms = 1;
    for (uint32_t i = 1; i <= XBITS; i++)
        totperms *= i;

    #pragma omp parallel for
    for (uint64_t bmask_idx = 0; bmask_idx < total_bmask; bmask_idx++)
    {
        uint64_t bankmask = bank_masks[bmask_idx];
        uint64_t total_rmask = 0;
        // allocations to be used (inside loop to make sure threads do not conflict)
        // --------------------------------------------------------------------
        uint64_t * xrow_masks = do_masked_combinations(n - BankBits, XBITS, &total_rmask, bankmask);
        uint32_t * xr_bits = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
        uint32_t * bb_bits = (uint32_t*)malloc(sizeof(uint32_t)*BankBits);
        uint32_t total_buckets = 1 << BankBits;
        uint64_t ** buckets = (uint64_t**)malloc(sizeof(uint64_t*)*total_buckets);
        uint32_t * bsizes = (uint32_t*)malloc(sizeof(uint32_t)*total_buckets);
        uint32_t * bsizes_alloc = (uint32_t*)malloc(sizeof(uint32_t)*total_buckets);
        uint64_t HE_set_size = rowbits + colbits - XBITS;
        uint64_t ** HE = (uint64_t**)malloc(sizeof(uint64_t*) * HE_set_size);
        uint64_t  * aseq = (uint64_t*)malloc(sizeof(uint64_t) * aseq_len);
        uint32_t ** perms = (uint32_t**)malloc(sizeof(uint32_t*)*totperms);
        for (uint32_t i = 0; i < totperms; i++)
        {
                perms[i] = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
        }
        uint32_t totalbrbits = BankBits+XBITS;
        uint32_t * brbits = (uint32_t*)malloc(sizeof(uint32_t)*(totalbrbits));
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
        // got the xrows as well.
        for (uint64_t xrmask_idx = 0; xrmask_idx < total_rmask; xrmask_idx++)
        {
			uint64_t xrmask = xrow_masks[xrmask_idx];
			create_permutations(bankmask, xrmask, xr_bits, bb_bits, perms, totperms);
            for (uint32_t perms_idx = 0; perms_idx < totperms; perms_idx++)
            {
                memset(bsizes, 0, sizeof(uint32_t)*total_buckets);
                for (uint32_t i = 0; i < total_buckets; i++)
                {
                    buckets[i] = (uint64_t*)malloc(sizeof(uint64_t)* (aseq_len + 1));
                    bsizes_alloc[i] = aseq_len + 1;
                }
				gen_buckets(buckets, bsizes, addresses, aseq_len, bb_bits, perms[perms_idx], BankBits);
                uint64_t * rowx_star;
                allocpool(&mpool, &rowx_star);
                gen_rowstar_edges(rowx_star, buckets, total_buckets, bsizes, aseq, aseq_len, xrmask);
                for (uint32_t i = 0; i < BankBits; i++)
                    brbits[i] = bb_bits[i];
                for (uint32_t i = BankBits; i < totalbrbits; i++)
                    brbits[i] = perms[perms_idx][i - BankBits];
				
                qsort(brbits, totalbrbits, sizeof(uint32_t), qs_cmpfunc_desc);
                rm_bucket_bits(buckets, total_buckets, bsizes, brbits, totalbrbits);
                // generate address sequence for the bank xor row bits selected in order defined by permidx
                gen_aseq(buckets, total_buckets, bsizes, aseq);

                // generate hyperedges for every relevant bit
                gen_edges(HE, aseq, aseq_len, HE_set_size, &mpool);
                BBSolBits aux = BB_iterative(HE, HE_set_size, rowbits - XBITS,
                                    cur_alpha, rowx_star, &mpool,
                                    SolBit_stack, SolSet_stack, NewSolBit_stack, NewSolSet_stack);
                aux.BankBits = bankmask;
                aux.RowXorBits = xrmask;
				//if (aux.SolBits != 0)
				//	printf("%lu\n",aux.SolBits);
                #pragma omp critical
                {
					update_solutions(&cur_alpha, aux, &curminpop, minbb, &minbb_size, perms,perms_idx);
				}
				for (uint64_t i = 0; i < HE_set_size; i++)
				{
					free_pool(&mpool, HE[i]);
				}
				free_pool(&mpool, rowx_star);
				for (uint32_t b = 0; b < total_buckets; b++)
				{
					free(buckets[b]);
				}
			}
		}
		// FREE EVERYONE!
		destroypool(&mpool);
		free(xrow_masks);
		free(xr_bits);
		free(bb_bits);
		free(buckets);
		free(bsizes);
		free(bsizes_alloc);
		free(HE);
		free(aseq);
		for (uint32_t i = 0; i < totperms; i++)
			free(perms[i]);
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


uint64_t
read_input_addresses(char * inputfile, uint64_t * A, uint64_t expected_lines)
{
	uint64_t totlines = 0;
	FILE * f = fopen(inputfile, "r");
	uint64_t * B = (uint64_t *)malloc(sizeof(uint64_t) * (expected_lines+10));
	uint64_t newaddr = 0;
	char dummy[3];
	uint64_t bufferlines = 0;
	uint64_t k = 32;  // predefined buffer size, aka high-water mark for memory controller.
	uint32_t prev = 0;  // 0 = read, 1 = write
	while (fscanf(f, "%s %lu\n", &dummy, &newaddr) != EOF)
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
					totlines++;
				}
				bufferlines = 0;
			}
			A[totlines] = newaddr >> BIT_IGNORE;
			totlines++;
			prev = 0;
		} else {
			if (prev)  // previous was a WRITE (W -> W)
			{
				A[totlines] = newaddr >> BIT_IGNORE;
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
						totlines++;
					}
					bufferlines = 0;
					A[totlines] = newaddr >> BIT_IGNORE;
					totlines++;
					prev = 1;
				}
			}
		}
	}
	for (uint32_t j = 0; j < bufferlines; j++)
	{
		A[totlines] = B[j] >> 8;
		totlines++;
	}
	bufferlines = 0;
	free(B);
	fclose(f);
	return totlines;
}

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
		printf("adjusted\n");
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
	uint64_t * A = (uint64_t *)malloc(sizeof(uint64_t) * (expected_lines+10));
	uint64_t totlines =	read_input_addresses(inputfile, A, expected_lines);
	uint64_t total_Sols = 0;
	BBSolBits * bbmin = solver(A, totlines, BANK_BITS, TOTAL_BITS - (BANK_BITS + COLROW_BITS + BIT_IGNORE), COLROW_BITS, &total_Sols);
	print_solution(bbmin, total_Sols);
	free(bbmin);
	free(A);
	return 0;
}
