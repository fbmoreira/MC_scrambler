#include<stdint.h>
#include<string.h>
#include<malloc.h>
#include<stdlib.h>


typedef struct info {
	uint64_t end_tstamp;
	uint32_t index;
	uint8_t type_weight;
} info;

typedef struct queue {
	info * buf;
	uint32_t size;
	uint32_t cursize;
	uint64_t ignored_reqs;
} queue;


static inline void 
init_queue( queue * Q, uint32_t size) 
{
	Q->size = size;
	Q->buf = (info*)malloc(sizeof(info)*size);
	memset(Q->buf, 0, sizeof(info)*size);
  Q->cursize = 0;
	Q->ignored_reqs = 0;
}

static inline void
destroy_queue(queue * Q) 
{
	free(Q->buf);
}

static inline uint32_t
empty(queue *Q)
{
	return (Q->cursize == 0);
}

static inline void
enqueue(queue * Q, info i)
{
	if (Q->cursize == Q->size) //full
	{
		Q->ignored_reqs++; // we should *really* penalize a mapping when this happens.
	} else {
		Q->buf[Q->cursize].index = i.index;
		Q->buf[Q->cursize].end_tstamp = i.end_tstamp;
		Q->buf[Q->cursize].type_weight = i.type_weight;
		Q->cursize++;
	}
}

static inline info
dequeue(queue * Q)
{
	if (Q->cursize == 0)
	{
		printf("error, dequeue on empty queue\n");
		exit(1);
	} else {
		info aux;
		aux.index = Q->buf[0].index;
		aux.end_tstamp = Q->buf[0].end_tstamp;
		aux.type_weight = Q->buf[0].type_weight;
		Q->cursize--;
		if (!empty(Q))
			memmove(&(Q->buf[0]),&(Q->buf[1]), sizeof(info)*Q->cursize);
		return aux;
	}
}

static inline void
print_queue(queue * Q)
{
	for (uint32_t i = 0; i < Q->cursize; i++)
	{
		printf("%u:  %u %lu %u\n", i, Q->buf[i].index, Q->buf[i].end_tstamp, Q->buf[i].type_weight);
	}
}
