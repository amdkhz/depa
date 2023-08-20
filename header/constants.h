#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#define FANOUT_RADIX 3//8
#define BUCKETS (1 << FANOUT_RADIX)
#define BLOCK_SIZE 4//509
#define BLOCK_BITS 2//9
#define DATA_SIZE_BYTE 8 // int64_t
#define DATA_SIZE_BIT (DATA_SIZE_BYTE << 3) //64 bits
#define CachBlockSize 4
#define MIN_BUCKET_SIZE 4
#define SAMPLE_SIZE 2//1000

#endif

