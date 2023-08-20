#ifndef SAMPLE_SIZE
#define SAMPLE_SIZE 4
#endif


#ifndef __ADAPTIVE_INDEX_H__
#define __ADAPTIVE_INDEX_H__

#include <vector>
#include <algorithm>
#include <map>
#include <immintrin.h>
#include <bitset>
#include <limits.h>
#include <iostream>

// #include <jemalloc/jemalloc.h>
#include <mimalloc.h>
#define LOG2(X) ((int) (DATA_SIZE_BIT - 1 - __builtin_clz((X))))
#define next_pow2(x) ((int) (x == 1 ? 1 : 1<<(DATA_SIZE_BIT - __builtin_clz(x-1))))

#include "common.h"
#include "../resource/ittnotify/include/ittnotify.h"

using namespace std;


class AdaptiveShiftIndex
{
    using FuncPtr = void (AdaptiveShiftIndex::*)(_Rb_tree_iterator<pair<const int32_t, Bucket>> &, int32_t, Timer &, ___itt_domain *);
private:
    Column *column;
    size_t totalBits;
    size_t fanoutShift;
    int32_t minValue;
    int32_t maxValue;
    int queryCounter;
    map<int32_t, Bucket> keyBucketMap;
    CacheLineBlock bucketsCacheLines[BUCKETS];

    inline void printIndex()
    {
        for (auto it = keyBucketMap.begin(); it != keyBucketMap.end(); it++){
            (it->second).printBucket(it->first);
        }
    }

    inline void merge(_Rb_tree_iterator<pair<const int32_t, Bucket > >& bucketIt, Timer& timer, ___itt_domain* domain){
        auto mergeTask = __itt_string_handle_create("merge Buckets");
        __itt_task_begin(domain, __itt_null, __itt_null, mergeTask);
        timer.start("merge");
        if((next(bucketIt, 1) != keyBucketMap.end()) && ((*(next(bucketIt, 1)->second).head.begin()).sorted == 1)){
            (bucketIt->second) = mergeBuckets(bucketIt->second, next(bucketIt, 1)->second);
            keyBucketMap.erase(next(bucketIt, 1));
        }
        else if ((bucketIt != keyBucketMap.begin()) && ((*(prev(bucketIt, 1)->second).head.begin()).sorted == 1)){
            prev(bucketIt, 1)->second = mergeBuckets(bucketIt->second, prev(bucketIt, 1)->second);
            keyBucketMap.erase(bucketIt++);
        }
        timer.end("merge");
        __itt_task_end(domain);
    }
        
    inline void distributeLog(_List_iterator<Block>& blockIt, array<Bucket, BUCKETS>& buckets,  Timer& timer, ___itt_domain* domain){
        auto distributeTask = __itt_string_handle_create("distributeLog Buckets");
        __itt_task_begin(domain, __itt_null, __itt_null, distributeTask);
        timer.start("distributeLog");

        if(blockIt->sorted){
            return;
        }
        auto data = (*blockIt).data.get();
        auto cnt = (*blockIt).count;

        #pragma unroll
            for (auto i = 0; i < cnt; i++)
            {
                int idx = LOG2(data[i]);
                bucketsCacheLines[idx].addElement(data[i], buckets[idx]);
            }
        
        timer.end("distributeLog");
        __itt_task_end(domain);
    }

    inline void distribute(_List_iterator<Block>& blockIt, int32_t delta, int32_t partitionFanoutMask, int32_t partitionFanoutShift, array<Bucket, BUCKETS>& buckets, Timer& timer, ___itt_domain* domain){
        auto distributeTask = __itt_string_handle_create("distribute Buckets");
        __itt_task_begin(domain, __itt_null, __itt_null, distributeTask);
        timer.start("distribute");

        if(blockIt->sorted){
            return;
        }
        auto data = (*blockIt).data.get();
        auto cnt = (*blockIt).count;

        // int cnt16 = closestMultiple(cnt, 16);
        // __m512i delta_ = _mm512_set_epi32(delta, delta, delta, delta, delta, delta, delta, delta, delta, delta, delta, delta, delta, delta, delta, delta);
        // __m512i partitionFanoutMask_ = _mm512_set_epi32(partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask, partitionFanoutMask);
        // __m512i partitionFanoutShift_ = _mm512_set_epi32(partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift, partitionFanoutShift);

        // #pragma unroll
        // for (auto i = 0; i < cnt16; i += 16){
        //     __m512i castedValues = _mm512_loadu_si512((__m512i*)(data+i));
        //     auto indices = _mm512_add_epi32(delta_, castedValues);
        //     indices = _mm512_and_si512(indices, partitionFanoutMask_);
        //     indices = _mm512_srlv_epi32(indices, partitionFanoutShift_);
        //     #pragma unroll
        //     for (auto j = 0; j < 16; j++)
        //     {
        //         auto idx = mm512_extract_epi32(indices, j);
        //         bucketsCacheLines[idx].addElement(data[i+j], buckets[idx]);
        //     }
        // }
        
        // #pragma unroll
        // for (auto i = cnt16; i < cnt; i++){
        //     int idx = ((data[i] + delta) & partitionFanoutMask) >> partitionFanoutShift;
        //     bucketsCacheLines[idx].addElement(data[i], buckets[idx]);
        // }

        #pragma unroll
            for (auto i = 0; i < cnt; i++)
            {
                int idx = ((data[i] + delta) & partitionFanoutMask) >> partitionFanoutShift;
                bucketsCacheLines[idx].addElement(data[i], buckets[idx]);
            }

        timer.end("distribute");
        __itt_task_end(domain);
    }

    inline void distributeLeftSkew(_List_iterator<Block>& blockIt, int32_t partitionFanoutMask, int32_t partitionFanoutShift, array<Bucket, BUCKETS>& buckets, Timer& timer, ___itt_domain* domain){
        auto distributeTask = __itt_string_handle_create("distribute left-skewed Buckets");
        __itt_task_begin(domain, __itt_null, __itt_null, distributeTask);
        timer.start("distributeLeftSkew");

        if(blockIt->sorted){
            return;
        }
        auto data = (*blockIt).data.get();
        auto cnt = (*blockIt).count;

        #pragma unroll
            for (auto i = 0; i < cnt; i++)
            {
                int idx = (int(pow(LOG2(data[i]), 3)) & partitionFanoutMask) >> partitionFanoutShift;
                bucketsCacheLines[idx].addElement(data[i], buckets[idx]);
            }
        
        timer.end("distributeLeftSkew");
        __itt_task_end(domain);
    }

    inline void distributeRightSkew(_List_iterator<Block>& blockIt, int32_t partitionFanoutMask, int32_t partitionFanoutShift, array<Bucket, BUCKETS>& buckets, Timer& timer, ___itt_domain* domain){
        auto distributeTask = __itt_string_handle_create("distribute right-skewed Buckets");
        __itt_task_begin(domain, __itt_null, __itt_null, distributeTask);
        timer.start("distributeRightSkew");

        if(blockIt->sorted){
            return;
        }
        auto data = (*blockIt).data.get();
        auto cnt = (*blockIt).count;

        #pragma unroll
            for (auto i = 0; i < cnt; i++)
            {
                int idx = (((data[i] * data[i]) * LOG2(data[i])) & partitionFanoutMask) >> partitionFanoutShift;
                bucketsCacheLines[idx].addElement(data[i], buckets[idx]);
            }
        
        timer.end("distributeRightSkew");
        __itt_task_end(domain);
    }

    inline void distributeNormal(_List_iterator<Block>& blockIt, int32_t partitionFanoutMask, int32_t partitionFanoutShift, int32_t meanVal, int32_t offset, array<Bucket, BUCKETS>& buckets, Timer& timer, ___itt_domain* domain){
        auto distributeTask = __itt_string_handle_create("distribute Normal Buckets");
        __itt_task_begin(domain, __itt_null, __itt_null, distributeTask);
        timer.start("distributeNormal");

        if(blockIt->sorted){
            return;
        }
        auto data = (*blockIt).data.get();
        auto cnt = (*blockIt).count;

        #pragma unroll
            for (auto i = 0; i < cnt; i++)
            {
                int idx = ((std::signbit(meanVal-data[i]) * int(pow(LOG2(abs(meanVal-data[i])), 3)) + offset) & partitionFanoutMask) >> partitionFanoutShift;
                bucketsCacheLines[idx].addElement(data[i], buckets[idx]);
            }
        
        timer.end("distributeNormal");
        __itt_task_end(domain);
    }

    inline double shiftPartitionSample(vector<int>& data, int pivot){
        auto bucketsFreq = vector<int>(BUCKETS, 0);
        auto minVal = *std::min_element(data.begin(), data.end());
        auto maxVal = *std::max_element(data.begin(), data.end());
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(minVal ^ maxVal));
        int partitionFanoutShift = dissimilarBit - FANOUT_RADIX;
        int bucketSize = 1 << max(0, (__builtin_clz(minVal) - __builtin_clz(maxVal) + 1 - FANOUT_RADIX));

        if(partitionFanoutShift > BLOCK_BITS && bucketSize > MIN_BUCKET_SIZE){
            int partitionFanoutMask = (((1 << dissimilarBit) - 1) ^ ((1 << partitionFanoutShift) - 1));

            int32_t delta = closestMultiple(pivot, bucketSize) - pivot;

            for(auto n : data){
                bucketsFreq[((n + delta) & partitionFanoutMask) >> partitionFanoutShift]++;
            }
        }
        return getVariance(bucketsFreq);
    }

    inline double logPartitionSample(vector<int>& data){
        auto bucketsFreq = vector<int>(BUCKETS, 0);
        auto minVal = max(LOG2(*std::min_element(data.begin(), data.end())), 0);
        auto maxVal = max(LOG2(*std::max_element(data.begin(), data.end())), 0);
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(minVal ^ maxVal));

        if(dissimilarBit > LOG2(CacheBlockSize)){
            for(auto n : data){
                bucketsFreq[LOG2(n)]++;
            }
        }
        return getVariance(bucketsFreq);
    }

    inline double leftSkewPartitionSample(vector<int>& data){
        auto bucketsFreq = vector<int>(BUCKETS, 0);
        auto minData = *std::min_element(data.begin(), data.end());
        auto maxData = *std::max_element(data.begin(), data.end());
        auto minVal = int(pow(LOG2(minData), 3));
        auto maxVal = int(pow(LOG2(maxData), 3));
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(minVal ^ maxVal));
        int partitionFanoutShift = dissimilarBit - FANOUT_RADIX;

        if(dissimilarBit > LOG2(CacheBlockSize)){
            for(auto n : data){
                bucketsFreq[int(pow(LOG2(n), 3)) >> partitionFanoutShift]++;
            }
        }
        return getVariance(bucketsFreq);
    }

    inline double rightSkewPartitionSample(vector<int>& data){
        auto bucketsFreq = vector<int>(BUCKETS, 0);
        auto minData = *std::min_element(data.begin(), data.end());
        auto maxData = *std::max_element(data.begin(), data.end());
        auto minVal = minData * minData * LOG2(minData);
        auto maxVal = maxData * maxData * LOG2(maxData);
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(minVal ^ maxVal));
        int partitionFanoutShift = dissimilarBit - FANOUT_RADIX;

        if(dissimilarBit > LOG2(CacheBlockSize)){
            for(auto n : data){
                bucketsFreq[(n * n  * LOG2(n)) >> partitionFanoutShift]++;
            }
        }
        return getVariance(bucketsFreq);
    }

    inline double normalPartitionSample(vector<int>& data){
        auto bucketsFreq = vector<int>(BUCKETS, 0);
        auto meanVal = int(getMean(data));
        int minData = *std::min_element(data.begin(), data.end());
        int maxData = *std::max_element(data.begin(), data.end());
        auto offset = meanVal - minData;
        auto minVal = int(pow(LOG2(meanVal - minData), 3)) + offset;
        auto maxVal = int(pow(LOG2(maxData - meanVal), 3)) + offset;
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(minVal ^ maxVal));
        int partitionFanoutShift = dissimilarBit - FANOUT_RADIX;

        if(dissimilarBit > LOG2(CacheBlockSize)){
            for(auto n : data){
                bucketsFreq[(std::signbit(meanVal-n) * int(pow(LOG2(abs(meanVal-n)), 3)) + offset) >> partitionFanoutShift]++;
            }
        }
        return getVariance(bucketsFreq);
    }


    inline void partition(_Rb_tree_iterator<pair<const int32_t, Bucket > >& bucketIt, int32_t pivot, Timer& timer, ___itt_domain* domain)
    {
        timer.start("partition");
        if (pivot < minValue || pivot > maxValue)
        {
            timer.end("partition");
            return;
        }

        int32_t partitionMaxVal;
        if ((bucketIt == keyBucketMap.end()) || (next(bucketIt, 1) == keyBucketMap.end())){
            partitionMaxVal = maxValue;
            if (bucketIt == keyBucketMap.end()){
                bucketIt--;
            }
        }
        else{
            partitionMaxVal = next(bucketIt, 1)->first;
        }
        int32_t partitionMinVal = bucketIt->first;
        timer.end("partition");

        if((*(bucketIt->second).head.begin()).sorted == 1 && next((bucketIt->second).head.begin(), 1) != (bucketIt->second).head.end()){
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            return;
        }

        if(next((bucketIt->second).head.begin(), 1) == (bucketIt->second).head.end() && keyBucketMap.size() > 1){
            if((*(bucketIt->second).head.begin()).sorted == 1){
                merge(bucketIt, timer, domain);
            }
            else{
                timer.start("sort");
                (bucketIt->second).sortBucket();
                timer.end("sort");
            }
            return;
        }

        timer.start("partition");
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(partitionMinVal ^ partitionMaxVal));
        int partitionFanoutShift = dissimilarBit - FANOUT_RADIX;
        int bucketSize = 1 << max(0, (__builtin_clz(partitionMinVal) - __builtin_clz(partitionMaxVal) + 1 - FANOUT_RADIX));

        if(partitionFanoutShift > BLOCK_BITS && bucketSize > MIN_BUCKET_SIZE){
            int partitionFanoutMask = (((1 << dissimilarBit) - 1) ^ ((1 << partitionFanoutShift) - 1));

            int32_t delta = closestMultiple(pivot, bucketSize) - pivot;

            array<Bucket, BUCKETS> buckets;
            timer.end("partition");
            for(auto it = (bucketIt->second).head.begin(); it != (bucketIt->second).head.end(); ++it){
                distribute(it, delta, partitionFanoutMask, partitionFanoutShift, buckets, timer, domain);
            }

            timer.start("partition");
            for (auto i = 0; i < BUCKETS; i++)
            {
                bucketsCacheLines[i].flush(buckets[i]);
                if ((*buckets[i].head.begin()).count)
                {
                    keyBucketMap[(i << partitionFanoutShift) | (bucketIt->first & (((1 << totalBits) -1) ^ ((1 << dissimilarBit) -1)))] = move(buckets[i]);
                }
            }
            timer.end("partition");
        }
        else{
            auto sortBucketTask = __itt_string_handle_create("Sort Buckets");
            __itt_task_begin(domain, __itt_null, __itt_null, sortBucketTask);
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            __itt_task_end(domain);
        }
    }

    inline void partitionLog(_Rb_tree_iterator<pair<const int32_t, Bucket > >& bucketIt, int32_t pivot, Timer& timer, ___itt_domain* domain)
    {
        timer.start("partition");
        if (pivot < minValue || pivot > maxValue)
        {
            timer.end("partition");
            return;
        }

        int32_t partitionMaxVal;
        if ((bucketIt == keyBucketMap.end()) || (next(bucketIt, 1) == keyBucketMap.end())){
            partitionMaxVal = LOG2(maxValue);
            if (bucketIt == keyBucketMap.end()){
                bucketIt--;
            }
        }
        else{
            partitionMaxVal = LOG2(next(bucketIt, 1)->first);
        }
        int32_t partitionMinVal = max(LOG2(bucketIt->first),0);
        timer.end("partition");

        if((*(bucketIt->second).head.begin()).sorted == 1 && next((bucketIt->second).head.begin(), 1) != (bucketIt->second).head.end()){
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            return;
        }

        if(next((bucketIt->second).head.begin(), 1) == (bucketIt->second).head.end() && keyBucketMap.size() > 1){
            if((*(bucketIt->second).head.begin()).sorted == 1){
                merge(bucketIt, timer, domain);
            }
            else{
                timer.start("sort");
                (bucketIt->second).sortBucket();
                timer.end("sort");
            }
            return;
        }

        timer.start("partition");
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(partitionMinVal ^ partitionMaxVal));
        if(dissimilarBit > LOG2(CacheBlockSize)){
            array<Bucket, BUCKETS> buckets;
            timer.end("partition");
            for(auto it = (bucketIt->second).head.begin(); it != (bucketIt->second).head.end(); ++it){
                distributeLog(it, buckets, timer, domain);
            }

            timer.start("partition");
            for (auto i = 0; i < BUCKETS; i++)
            {
                bucketsCacheLines[i].flush(buckets[i]);
                if ((*buckets[i].head.begin()).count)
                {
                    keyBucketMap[1 << i] = move(buckets[i]);
                }
            }
            timer.end("partition");
        }
        else{
            auto sortBucketTask = __itt_string_handle_create("Sort Buckets");
            __itt_task_begin(domain, __itt_null, __itt_null, sortBucketTask);
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            __itt_task_end(domain);
        }
    }

    inline void partitionLeftSkew(_Rb_tree_iterator<pair<const int32_t, Bucket > >& bucketIt, int32_t pivot, Timer& timer, ___itt_domain* domain)
    {
        timer.start("partition");
        if (pivot < minValue || pivot > maxValue)
        {
            timer.end("partition");
            return;
        }

        int32_t partitionMaxVal;
        if ((bucketIt == keyBucketMap.end()) || (next(bucketIt, 1) == keyBucketMap.end())){
            partitionMaxVal = max(int(pow(LOG2(maxValue), 3)), 0);
            if (bucketIt == keyBucketMap.end()){
                bucketIt--;
            }
        }
        else{
            partitionMaxVal = max(int(pow(LOG2(next(bucketIt, 1)->first), 3)), 0);
        }
        int32_t partitionMinVal = max(int(pow(LOG2(bucketIt->first), 3)), 0);
        timer.end("partition");

        if((*(bucketIt->second).head.begin()).sorted == 1 && next((bucketIt->second).head.begin(), 1) != (bucketIt->second).head.end()){
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            return;
        }

        if(next((bucketIt->second).head.begin(), 1) == (bucketIt->second).head.end() && keyBucketMap.size() > 1){
            if((*(bucketIt->second).head.begin()).sorted == 1){
                merge(bucketIt, timer, domain);
            }
            else{
                timer.start("sort");
                (bucketIt->second).sortBucket();
                timer.end("sort");
            }
            return;
        }

        timer.start("partition");
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(partitionMinVal ^ partitionMaxVal));
        int partitionFanoutShift = dissimilarBit - FANOUT_RADIX;
        int bucketSize = 1 << max(0, (__builtin_clz(partitionMinVal) - __builtin_clz(partitionMaxVal) + 1 - FANOUT_RADIX));

        if(partitionFanoutShift > BLOCK_BITS && bucketSize > MIN_BUCKET_SIZE){
            int partitionFanoutMask = (((1 << dissimilarBit) - 1) ^ ((1 << partitionFanoutShift) - 1));

            array<Bucket, BUCKETS> buckets;
            timer.end("partition");
            for(auto it = (bucketIt->second).head.begin(); it != (bucketIt->second).head.end(); ++it){
                distributeLeftSkew(it, partitionFanoutMask, partitionFanoutShift, buckets, timer, domain);
            }

            timer.start("partition");
            for (auto i = 0; i < BUCKETS; i++)
            {
                bucketsCacheLines[i].flush(buckets[i]);
                if ((*buckets[i].head.begin()).count)
                {
                    keyBucketMap[(i << partitionFanoutShift) | (bucketIt->first & (((1 << totalBits) -1) ^ ((1 << dissimilarBit) -1)))] = move(buckets[i]);
                }
            }
            timer.end("partition");
        }
        else{
            auto sortBucketTask = __itt_string_handle_create("Sort Buckets");
            __itt_task_begin(domain, __itt_null, __itt_null, sortBucketTask);
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            __itt_task_end(domain);
        }
    }

    inline void partitionRightSkew(_Rb_tree_iterator<pair<const int32_t, Bucket > >& bucketIt, int32_t pivot, Timer& timer, ___itt_domain* domain)
    {
        timer.start("partition");
        if (pivot < minValue || pivot > maxValue)
        {
            timer.end("partition");
            return;
        }

        int32_t partitionMaxVal;
        if ((bucketIt == keyBucketMap.end()) || (next(bucketIt, 1) == keyBucketMap.end())){
            partitionMaxVal = max(maxValue * maxValue * LOG2(maxValue), 0);
            if (bucketIt == keyBucketMap.end()){
                bucketIt--;
            }
        }
        else{
            partitionMaxVal = max((next(bucketIt, 1)->first) * (next(bucketIt, 1)->first) * LOG2(next(bucketIt, 1)->first), 0);
        }
        int32_t partitionMinVal = max((bucketIt->first) * (bucketIt->first) * LOG2(bucketIt->first), 0);
        timer.end("partition");

        if((*(bucketIt->second).head.begin()).sorted == 1 && next((bucketIt->second).head.begin(), 1) != (bucketIt->second).head.end()){
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            return;
        }

        if(next((bucketIt->second).head.begin(), 1) == (bucketIt->second).head.end() && keyBucketMap.size() > 1){
            if((*(bucketIt->second).head.begin()).sorted == 1){
                merge(bucketIt, timer, domain);
            }
            else{
                timer.start("sort");
                (bucketIt->second).sortBucket();
                timer.end("sort");
            }
            return;
        }

        timer.start("partition");
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(partitionMinVal ^ partitionMaxVal));
        int partitionFanoutShift = dissimilarBit - FANOUT_RADIX;
        int bucketSize = 1 << max(0, (__builtin_clz(partitionMinVal) - __builtin_clz(partitionMaxVal) + 1 - FANOUT_RADIX));

        if(partitionFanoutShift > BLOCK_BITS && bucketSize > MIN_BUCKET_SIZE){
            int partitionFanoutMask = (((1 << dissimilarBit) - 1) ^ ((1 << partitionFanoutShift) - 1));

            array<Bucket, BUCKETS> buckets;
            timer.end("partition");
            for(auto it = (bucketIt->second).head.begin(); it != (bucketIt->second).head.end(); ++it){
                distributeRightSkew(it, partitionFanoutMask, partitionFanoutShift, buckets, timer, domain);
            }

            timer.start("partition");
            for (auto i = 0; i < BUCKETS; i++)
            {
                bucketsCacheLines[i].flush(buckets[i]);
                if ((*buckets[i].head.begin()).count)
                {
                    keyBucketMap[(i << partitionFanoutShift) | (bucketIt->first & (((1 << totalBits) -1) ^ ((1 << dissimilarBit) -1)))] = move(buckets[i]);
                }
            }
            timer.end("partition");
        }
        else{
            auto sortBucketTask = __itt_string_handle_create("Sort Buckets");
            __itt_task_begin(domain, __itt_null, __itt_null, sortBucketTask);
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            __itt_task_end(domain);
        }
    }

    inline void partitionNormal(_Rb_tree_iterator<pair<const int32_t, Bucket > >& bucketIt, int32_t pivot, Timer& timer, ___itt_domain* domain)
    {
        timer.start("partition");
        if (pivot < minValue || pivot > maxValue)
        {
            timer.end("partition");
            return;
        }

        int32_t partitionMaxVal;
        if ((bucketIt == keyBucketMap.end()) || (next(bucketIt, 1) == keyBucketMap.end())){
            partitionMaxVal = maxValue;
            if (bucketIt == keyBucketMap.end()){
                bucketIt--;
            }
        }
        else{
            partitionMaxVal = next(bucketIt, 1)->first;
        }
        int32_t partitionMinVal = bucketIt->first;
        timer.end("partition");

        auto meanVal = int((partitionMinVal + partitionMaxVal)/2);
        auto offset = (meanVal - partitionMinVal);
        partitionMinVal = int(pow(LOG2(meanVal), 3)) + offset;
        partitionMaxVal = int(pow(LOG2(partitionMaxVal), 3)) + offset;

        if((*(bucketIt->second).head.begin()).sorted == 1 && next((bucketIt->second).head.begin(), 1) != (bucketIt->second).head.end()){
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            return;
        }

        if(next((bucketIt->second).head.begin(), 1) == (bucketIt->second).head.end() && keyBucketMap.size() > 1){
            if((*(bucketIt->second).head.begin()).sorted == 1){
                merge(bucketIt, timer, domain);
            }
            else{
                timer.start("sort");
                (bucketIt->second).sortBucket();
                timer.end("sort");
            }
            return;
        }

        timer.start("partition");
        int dissimilarBit = (DATA_SIZE_BIT - __builtin_clz(partitionMinVal ^ partitionMaxVal));
        int partitionFanoutShift = dissimilarBit - FANOUT_RADIX;
        int bucketSize = 1 << max(0, (__builtin_clz(partitionMinVal) - __builtin_clz(partitionMaxVal) + 1 - FANOUT_RADIX));

        if(partitionFanoutShift > BLOCK_BITS && bucketSize > MIN_BUCKET_SIZE){
            int partitionFanoutMask = (((1 << dissimilarBit) - 1) ^ ((1 << partitionFanoutShift) - 1));

            array<Bucket, BUCKETS> buckets;
            timer.end("partition");
            for(auto it = (bucketIt->second).head.begin(); it != (bucketIt->second).head.end(); ++it){
                distributeNormal(it, partitionFanoutMask, partitionFanoutShift, meanVal, offset, buckets, timer, domain);
            }

            timer.start("partition");
            for (auto i = 0; i < BUCKETS; i++)
            {
                bucketsCacheLines[i].flush(buckets[i]);
                if ((*buckets[i].head.begin()).count)
                {
                    keyBucketMap[(i << partitionFanoutShift) | (bucketIt->first & (((1 << totalBits) -1) ^ ((1 << dissimilarBit) -1)))] = move(buckets[i]);
                }
            }
            timer.end("partition");
        }
        else{
            auto sortBucketTask = __itt_string_handle_create("Sort Buckets");
            __itt_task_begin(domain, __itt_null, __itt_null, sortBucketTask);
            timer.start("sort");
            (bucketIt->second).sortBucket();
            timer.end("sort");
            __itt_task_end(domain);
        }
    }

    inline FuncPtr selectPartitionFunction(_Rb_tree_iterator<pair<const int32_t, Bucket > >& bucketIt, int pivot){
        FuncPtr partitionFunction = &AdaptiveShiftIndex::partition;
        auto partitionVar = INT_MAX;
        auto samples = std::vector<int>(SAMPLE_SIZE, 0);

        std::random_device dev;
        std::mt19937 rng(dev());

        if (bucketIt == keyBucketMap.end()){
            bucketIt--;
        }

        auto blockSamples = (int)(SAMPLE_SIZE / (bucketIt->second).head.size());
        for(auto blockIt = (bucketIt->second).head.begin(); blockIt != (bucketIt->second).head.end(); blockIt++){
            for (auto s = 0; s < blockSamples; s++){
                std::uniform_int_distribution<std::mt19937::result_type> dist(0, blockIt->count);
                samples.emplace_back(blockIt->data[dist(rng)]);
            }
        }
            
        auto shiftPartitionVar = shiftPartitionSample(samples, pivot);
        auto logPartitionVar = logPartitionSample(samples);
        auto leftSkewPartitionVar = leftSkewPartitionSample(samples);
        auto rightSkewPartitionVar = rightSkewPartitionSample(samples);
        auto normalPartitionVar = normalPartitionSample(samples);
        if (shiftPartitionVar < partitionVar){
            partitionVar = shiftPartitionVar;
            partitionFunction = &AdaptiveShiftIndex::partition;
        }
        if (logPartitionVar < partitionVar){
            partitionVar = logPartitionVar;
            partitionFunction = &AdaptiveShiftIndex::partitionLog;
        }
        if (leftSkewPartitionVar < partitionVar){
            partitionVar = leftSkewPartitionVar;
            partitionFunction = &AdaptiveShiftIndex::partitionLeftSkew;
        }
        if (rightSkewPartitionVar < partitionVar){
            partitionVar = rightSkewPartitionVar;
            partitionFunction = &AdaptiveShiftIndex::partitionRightSkew;
        }
        if (normalPartitionVar < partitionVar){
            partitionVar = normalPartitionVar;
            partitionFunction = &AdaptiveShiftIndex::partitionNormal;
        }

        return partitionFunction;
    }

    public:
        AdaptiveShiftIndex(Column* c, struct Timer& timer, ___itt_domain* domain) : column(c), totalBits(DATA_SIZE_BIT - __builtin_clz(c->maxValue)), fanoutShift(totalBits - FANOUT_RADIX), minValue(c->minValue), maxValue(c->maxValue)
        {
        queryCounter = 0;
        // partitionFunctionMap["Uniform Distribution"] = bind(&AdaptiveShiftIndex::partition, this, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4);
        // partitionFunctionMap["Exponential Distribution"] = bind(&AdaptiveShiftIndex::partition, this, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4);
        // partitionFunctionMap["Other Distribution"] = bind(&AdaptiveShiftIndex::partition, this, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4);
        // partitionFunctionMap["Skewed Distribution"] = bind(&AdaptiveShiftIndex::partitionLog, this, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4);
        // partitionFunctionMap["Spike Distribution"] = bind(&AdaptiveShiftIndex::partitionLog, this, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4);
        // partitionFunctionMap["Normal Distribution"] = bind(&AdaptiveShiftIndex::partitionLog, this, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4);

        auto initializationTask = __itt_string_handle_create("Initialization");
        __itt_task_begin(domain, __itt_null, __itt_null, initializationTask);

        timer.start("initialization");

        int32_t bucketSize = 1 << (__builtin_clz(minValue) - __builtin_clz(maxValue) + 1 - FANOUT_RADIX);
        int32_t delta = closestMultiple(minValue, bucketSize) - minValue;
        int32_t minBucket = (minValue + delta) >> fanoutShift;
        keyBucketMap[minBucket] = move(c->data);

        timer.end("initialization");

        __itt_task_end(domain);
        }

        inline auto rangeQuery(int32_t low, int32_t high, struct Timer& timer, ___itt_domain* domain)
        {
        queryCounter++;
        auto lowKeyLookupTask = __itt_string_handle_create("Lookup low key in tree");
        __itt_task_begin(domain, __itt_null, __itt_null, lowKeyLookupTask);

        // printIndex();
        timer.start("traverse");
        auto lowBucketIt = keyBucketMap.upper_bound(low);
        while (lowBucketIt != keyBucketMap.begin() && lowBucketIt->first > low)
        {
            lowBucketIt--;
            }
            timer.end("traverse");

            __itt_task_end(domain);

            auto firstPartitionTask = __itt_string_handle_create("First Partition");
            __itt_task_begin(domain, __itt_null, __itt_null, firstPartitionTask);

            partition(lowBucketIt, low, timer, domain);
            // partitionLog(lowBucketIt, low, timer, domain);
            // partitionFunctionMap[column->distribution](lowBucketIt, low, timer, domain);
            // printIndex();
            
            // (this->*selectPartitionFunction(lowBucketIt, low))(lowBucketIt, low, timer, domain);

            __itt_task_end(domain);

            timer.start("traverse");
            auto highKeyLookupTask = __itt_string_handle_create("Lookup high key in tree");
            __itt_task_begin(domain, __itt_null, __itt_null, highKeyLookupTask);
            
            auto highBucketIt = keyBucketMap.lower_bound(high);
            while(highBucketIt != keyBucketMap.begin() && highBucketIt->first > high){
                highBucketIt--;
            }
            timer.end("traverse");

            __itt_task_end(domain);

            auto secondPartitionTask = __itt_string_handle_create("Second Partition");
            __itt_task_begin(domain, __itt_null, __itt_null, secondPartitionTask);

            partition(highBucketIt, high, timer, domain);
            // partitionLog(highBucketIt, high, timer, domain);
            // partitionFunctionMap[column->distribution](highBucketIt, high, timer, domain);
            // (this->*selectPartitionFunction(highBucketIt, high))(highBucketIt, low, timer, domain);
            // printIndex();

            __itt_task_end(domain);

            auto sorted = 0.0;
            auto total = 0.0;
            for (auto it = keyBucketMap.begin(); it != keyBucketMap.end(); it++)
            {
                for (auto bit = it->second.head.begin(); bit != it->second.head.end(); bit++){
                total++;
                if ((*bit).sorted == 0)
                {
                    sorted++;
                }
                }
            }
            cout << sorted / total << endl;

                return make_pair(keyBucketMap.lower_bound(low), keyBucketMap.lower_bound(high));
        }

        inline size_t countResults(int32_t low, int32_t high, Timer& timer, ___itt_domain* domain){
            auto countResultTask = __itt_string_handle_create("Count results");
            __itt_task_begin(domain, __itt_null, __itt_null, countResultTask);
        
            timer.start("count");

            auto result = 0;
            auto lowBucketIt = keyBucketMap.upper_bound(low);
            while (lowBucketIt != keyBucketMap.begin() && lowBucketIt->first > low)
            {
                // to cover bucket that contains values equal to low
                lowBucketIt--;
            }
            
            auto highBucketIt = keyBucketMap.lower_bound(high);

            for (auto it = lowBucketIt; it != highBucketIt; it++){
                auto currMin = it->first;
                auto nxtMin = next(it, 1) != keyBucketMap.end() ? next(it, 1)->first : column->maxValue;
                if (currMin >= low && nxtMin < high)
                {
                    for(auto blockIt = (it->second).head.begin(); blockIt != (it->second).head.end(); ++blockIt){
                        result += (*blockIt).count;
                    }
                }
                else{
                    for(auto blockIt = (it->second).head.begin(); blockIt != (it->second).head.end(); ++blockIt){
                        auto indices = (*blockIt).lookupKeyRange(low, high);
                        if ((*blockIt).sorted)
                        {
                            result += (indices.second - indices.first);
                        }
                        else{
                            auto data = (*blockIt).data.get();
                            for (auto i = indices.first; i <= indices.second; i++)
                            {
                                if (data[i] >= low && data[i] < high)
                                {
                                    result++;
                                }
                            }
                        }
                    }
                }
            }

            timer.end("count");
            __itt_task_end(domain);

            return result;
        }
};


#endif
