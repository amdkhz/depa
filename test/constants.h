#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#define FANOUT_RADIX 3
#define BUCKETS (1 << FANOUT_RADIX)
#define BLOCK_SIZE 4
#define BLOCK_BITS 2
#define DATA_SIZE_BYTE 4
#define DATA_SIZE_BIT (DATA_SIZE_BYTE << 3)
#define CachBlockSize 4
#define MIN_BUCKET_SIZE 4


#endif