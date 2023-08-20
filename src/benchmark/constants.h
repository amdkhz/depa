#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__


#define FANOUT_RADIX 9
#define BUCKETS (1 << FANOUT_RADIX)
#define BLOCK_SIZE 4096//65536//131072//262144
#define BLOCK_BITS 12//16//17//18
#define DATA_SIZE_BYTE 4
#define DATA_SIZE_BIT (DATA_SIZE_BYTE << 3)
#define CacheBlockSize 16
#define MIN_BUCKET_SIZE 512
#define SAMPLE_SIZE 1000

#endif