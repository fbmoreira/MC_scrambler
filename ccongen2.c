/* Copyright 2021 Francis Birck Moreira GPL */

#include<malloc.h>
#include<immintrin.h>
#include<stdint.h>
#include "../MC_scrambler/pool_allocator.h"
#include "../MC_scrambler/avx_func.h"

#define MIN_BUF 512
#define XBITS 3
#define BIT_IGNORE 8
typedef struct bbresbits {
    uint64_t Fbits;
    uint64_t alpha;
    uint64_t bankbits;
    uint64_t rxbits;
	uint32_t perm[XBITS];
} bbresbits;

static inline bbresbits
BB_iterative(uint64_t ** HE, uint64_t HElen, uint64_t k,
        uint64_t upperbound, uint64_t * xored_rows_bitmap, mempool * pool,
        uint64_t chunklen, uint64_t * Fbit_stack, uint64_t ** Fset_stack,
        uint64_t * newbit_fs, uint64_t ** newset_fs)
{
    uint64_t fss = 0;  // Fset stack size

    bbresbits min;
    min.alpha = upperbound;
    min.Fbits = 0;
    allocpool(pool, &(Fset_stack[fss]));
  //  printf("prior to first call\n");
	bmset_union(Fset_stack[fss], xored_rows_bitmap, pool->chunklen);
    Fbit_stack[fss] = 0;
    fss++;  // we now have 1 element on the stack

// now we have a base F with a single hedge that has xrow's flips, we must find hedge sets of size k that have the minimum bit flips.
    for (uint64_t i = 0; (i < HElen && fss > 0); i++)
    {
        uint64_t nfs = 0;
        for (int64_t j = fss-1; j >= 0; j--)
        {
            if (((HElen - i) <= k) - (_mm_popcnt_u64(Fbit_stack[j])))  // bound: must add every edge to this F, so no point in adding a new F. just add k to the existent one, calc and quit.
            {
                Fbit_stack[j] |= (1 << i);
//				printf("prior to second call\n");
                bmset_union(Fset_stack[j], HE[i], pool->chunklen);
                if ((uint64_t)_mm_popcnt_u64(Fbit_stack[j]) == k)
                {
                    uint64_t newlen = bmset_popcnt(Fset_stack[j], pool->chunklen);
                    uint32_t a = (newlen < min.alpha);  // branchless technique here, we had awful performance prior to this.
                    min.alpha = (a * newlen) + ((!a) * min.alpha);
                    min.Fbits = (a * Fbit_stack[j]) + ((!a) * min.Fbits);
                    // delete Fset_stack[j];
                    free_pool(pool, Fset_stack[j]);
                    fss--;  // afterall we just deleted someone.
                }
            } else {  // finally, if we have an F where we have options
                allocpool(pool, &(newset_fs[nfs]));
//				printf("prior to third call\n");
                bmset_union(newset_fs[nfs], Fset_stack[j], pool->chunklen);
                newbit_fs[nfs] = Fbit_stack[j] | (1 << i);
//				printf("prior to fourth call\n");
                bmset_union(newset_fs[nfs], HE[i], pool->chunklen);
                uint64_t newlen = bmset_popcnt(newset_fs[nfs], pool->chunklen);
                if (newlen < min.alpha) {
                    if ((uint64_t)_mm_popcnt_u64(newbit_fs[nfs]) == k)
                    {
                        min.alpha = newlen;
                        min.Fbits = newbit_fs[nfs];
                        free_pool(pool, newset_fs[nfs]);
                    } else {  // respects upperbound but still does not have k elements.
                        nfs++;  // we effectively added an element
                    }
                } else {
                    free_pool(pool, newset_fs[nfs]);  // no need to keep the trash above our best upperbound so far.
                }
            }
        }
        // add all new guys to end of stack
        for (uint64_t ni = 0; ni < nfs; ni++)
        {
            Fbit_stack[fss] = newbit_fs[ni];  // add all new guys to end of stack
            Fset_stack[fss] = newset_fs[ni];
            // take possession
            take_ownership(pool, &(Fset_stack[fss]), &(newset_fs[ni]));
            fss++;
        }
    }
    // cleanup
    for (uint32_t i = 0; i < fss; i++)
    {
        free_pool(pool, Fset_stack[i]);
    }
    return min;
}

static inline uint64_t
remove_n_bit(uint64_t addr, uint64_t n)
{
    uint64_t upper = addr & 0xFFFFFFFE << n;
    uint64_t lower = addr & ((1 << n) - 1);
    return (upper >> 1) | lower;
}

static inline void
gen_buckets(uint64_t ** buckets, uint32_t total_buckets, uint32_t * bsizes, uint32_t * bsizes_alloc, uint64_t * addr, uint64_t aseq_len, uint32_t * masks1, uint32_t * masks2, uint32_t num_bits)
{
    for (uint64_t i = 0; i < aseq_len; i++)
    {
        uint64_t buck = 0;
        for (uint64_t j = 0; j < num_bits; j++)
        {
            buck  += ((addr[i] & masks1[j]) ^ (addr[i] & masks2[j]) ? 1 : 0) << j;
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
			//the worst branch, as per perf.
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

bbresbits *
solver(uint64_t * addresses, uint64_t aseq_len, uint32_t bankbits, uint32_t rowbits, uint32_t colbits, uint64_t * total_solutions)
{
    uint32_t n = bankbits + rowbits + colbits;
    uint64_t total_bmask = 0;
    uint64_t * bank_masks = do_masked_combinations(n, bankbits, &total_bmask, 0);
	uint64_t minbb_size = MIN_BUF;
    bbresbits * minbb = (bbresbits*)malloc(sizeof(bbresbits)*minbb_size);
	uint32_t curminpop = 0;
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
        uint64_t * xrow_masks = do_masked_combinations(n, XBITS, &total_rmask, bankmask);
        uint32_t * xr_bits = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
        uint32_t * bb_bits = (uint32_t*)malloc(sizeof(uint32_t)*bankbits);
        uint32_t total_buckets = 1 << bankbits;
        uint64_t ** buckets = (uint64_t**)malloc(sizeof(uint64_t*)*total_buckets);
        uint32_t * bsizes = (uint32_t*)malloc(sizeof(uint32_t)*total_buckets);
        uint32_t * bsizes_alloc = (uint32_t*)malloc(sizeof(uint32_t)*total_buckets);
        uint64_t HElen = rowbits + colbits - XBITS;
        uint64_t ** HE = (uint64_t**)malloc(sizeof(uint64_t*) * HElen);
        uint64_t  * aseq = (uint64_t*)malloc(sizeof(uint64_t) * aseq_len);
        uint32_t ** perms = (uint32_t**)malloc(sizeof(uint32_t*)*totperms);
        for (uint32_t i = 0; i < totperms; i++)
        {
                perms[i] = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
        }
        uint32_t totalbrbits = bankbits+XBITS;
        uint32_t * brbits = (uint32_t*)malloc(sizeof(uint32_t)*(totalbrbits));
        // ---------------------------------------------------------------------
        // bb_iterative mallocs
        uint64_t * Fbit_stack = malloc(sizeof(uint64_t)* (1 << 10));
        uint64_t ** Fset_stack = malloc(sizeof(uint64_t*)* (1 << 10));
        uint64_t * newbit_fs = (uint64_t*)malloc(sizeof(uint64_t) * (1 << 10));
        uint64_t ** newset_fs = (uint64_t**)malloc(sizeof(uint64_t) * (1 << 10));
        // ---------------------------------------------------------------------
        // Allocate a memory pool for the thread to avoid malloc/free
        mempool mpool;
        initpool(&mpool, aseq_len, 64);  // predefined estimate, might be reallocated
        // got the xrows as well.
        for (uint64_t xrmask_idx = 0; xrmask_idx < total_rmask; xrmask_idx++)
        {
            uint64_t xrmask = xrow_masks[xrmask_idx];
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

            for (uint32_t perms_idx = 0; perms_idx < totperms; perms_idx++)
            {
                memset(bsizes, 0, sizeof(uint32_t)*total_buckets);
                for (uint32_t i = 0; i < total_buckets; i++)
                {
                    buckets[i] = (uint64_t*)malloc(sizeof(uint64_t)* (aseq_len + 1));
                    bsizes_alloc[i] = aseq_len + 1;
                }
                gen_buckets(buckets, total_buckets, bsizes, bsizes_alloc, addresses, aseq_len, bb_bits, xr_bits, bankbits);
                uint64_t * rowx_star;
                allocpool(&mpool, &rowx_star);
                gen_rowstar_edges(rowx_star, buckets, total_buckets, bsizes, aseq, aseq_len, xrmask);
                for (uint32_t i = 0; i < bankbits; i++)
                    brbits[i] = bb_bits[i];
                for (uint32_t i = bankbits; i < totalbrbits; i++)
                    brbits[i] = perms[perms_idx][i - bankbits];
                qsort(brbits, totalbrbits, sizeof(uint32_t), qs_cmpfunc_desc);
                rm_bucket_bits(buckets, total_buckets, bsizes, brbits, totalbrbits);
                // generate address sequence for the bank xor row bits selected in order defined by permidx
                gen_aseq(buckets, total_buckets, bsizes, aseq);
                // generate hyperedges for every relevant bit
                gen_edges(HE, aseq, aseq_len, HElen, &mpool);
                bbresbits aux = BB_iterative(HE, HElen, rowbits - XBITS,
                                    cur_alpha, rowx_star, &mpool, aseq_len,
                                    Fbit_stack, Fset_stack, newbit_fs, newset_fs);
                aux.bankbits = bankmask;
                aux.rxbits = xrmask;
                #pragma omp critical
                {
                    if (cur_alpha > aux.alpha)  // eliminate previous solutions, this is better
                    {
                        printf("changed minbb, alpha is %lu, xrmask is %lu, bankmask is %lu\n", aux.alpha, aux.rxbits, aux.bankbits);
						// Erase previous "mins", we found a better solution.
						for (uint32_t mini = 0; mini < curminpop; mini++)
						{
//							free(minbb[mini].perm);
							minbb[mini].alpha = aseq_len;
						}
						// Set new first minbb
						minbb[0] = aux;
					//	minbb[0].perm = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
						memcpy(minbb[0].perm, perms[perms_idx], XBITS * sizeof(uint32_t));
						curminpop = 1;
						cur_alpha = aux.alpha;
                    } else {
						if ((cur_alpha == aux.alpha) && (aux.Fbits != 0))  // must add new valid solution
						{
							printf("adding solution, current number %u\n", curminpop);
							minbb[curminpop] = aux;
						//	minbb[curminpop].perm = (uint32_t*)malloc(sizeof(uint32_t)*XBITS);
							memcpy(minbb[curminpop].perm, perms[perms_idx], XBITS * sizeof(uint32_t));
							curminpop++;
							if (curminpop >= minbb_size)
							{
								printf("reallocating\n");
								minbb_size <<= 0x1ULL;
								minbb = (bbresbits*)realloc(minbb, minbb_size);
								if (minbb == NULL)
								{
									printf("failed to realloc min\n");
									exit(0);
								}
							}
						}
					}
				}
				for (uint64_t i = 0; i < HElen; i++)
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
		free(Fset_stack);
		free(Fbit_stack);
		free(newbit_fs);
		free(newset_fs);
	}
	free(bank_masks);
	*total_solutions = curminpop;
	return minbb;
}

	int main(int argc, char ** argv)
	{
		if (argc != 3)
		{
			printf("wrong number of args: %d\n", argc);
			for (uint32_t i = 0; i < argc; i++)
				printf("got %s\n", argv[i]);
			exit(0);
		}

		FILE * f = fopen(argv[1], "r");
		uint64_t expected_lines = strtoul(argv[2], NULL, 10);
		uint64_t * A = (uint64_t *)malloc(sizeof(uint64_t) * (expected_lines+10));
		uint64_t * B = (uint64_t *)malloc(sizeof(uint64_t) * (expected_lines+10));
		uint64_t newaddr = 0;
		char dummy[3];
		uint64_t totlines = 0;
		uint64_t bufferlines = 0;
		uint64_t k = 32;  // predefined buffer size, aka high-water mark for memory controller.
		uint32_t prev = 0;  // 0 = read, 1 = write
	///////////////////////////////////////////////////////
		// preprocess algorithm while reading
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
							bufferlines = 0;
							A[totlines] = newaddr >> BIT_IGNORE;
							totlines++;
							prev = 1;
						}
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
	////////////////////////////////////////////////////
		uint64_t total_solutions = 0;
		bbresbits * bbmin = solver(A, totlines, 3, 12, 7, &total_solutions);
		for (uint64_t i = 0; i < total_solutions; i++)
		{
			printf("bbmin.alpha %lu\n", bbmin[i].alpha);
			printf("got to end, bbmin Fbits = %lu\n", bbmin[i].Fbits);
			for (uint64_t i = 0; i < 64; i++)
			{
				if (1 & (bbmin[i].Fbits >> i))
					printf("%lu ", i);
			}
			printf("\n");
			printf("adjusted\n");
			uint64_t totmask = bbmin[i].bankbits | bbmin[i].rxbits;
			uint64_t curpos = __builtin_ctzll(~totmask);
			for (uint64_t i = 0; i < 64; i++)
			{
				if (1 & (bbmin[i].Fbits >> i))
				{
					printf("%lu ", i+curpos);
					totmask >>= curpos;
					curpos = __builtin_ctzll(~totmask);
				}
			}
			printf("\n");
			printf(" bank mask was %lu\n", bbmin[i].bankbits);
			printf(" xor rows msk was %lu\n", bbmin[i].rxbits);
		}
		free(bbmin);
		free(A);
		free(B);
		fclose(f);
		return 0;
}
