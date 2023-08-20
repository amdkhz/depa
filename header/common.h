#if !defined(CacheBlockSize)
#define CacheBlockSize 4
#endif

#if defined(__GNUC__)

int _mm512_cvtsi512_si32(__m512i a)
{
    __v16si b = (__v16si) a;
    return b[0];
}

#endif

#ifndef _COMMON_H_
#define _COMMOM_H_


#include <vector>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <cstring>
#include <omp.h>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>
#include <queue>
#include <functional>
#include <memory>
#include <list>
#include <sys/mman.h>
#include <random>
#include <cmath>      // for sqrt and fabs

#include "../resource/ska_sort.hpp"
// #include "../resource/regions-sort/RadixSort/radixSort.h"

// Thanks to https://stackoverflow.com/questions/9877700/getting-max-value-in-a-m128i-vector-with-sse
// int horizontal_max_Vec4i(__m128i x) {
//     __m128i max1 = _mm_shuffle_epi32(x, _MM_SHUFFLE(0,0,3,2));
//     __m128i max2 = _mm_max_epi32(x,max1);
//     __m128i max3 = _mm_shuffle_epi32(max2, _MM_SHUFFLE(0,0,0,1));
//     __m128i max4 = _mm_max_epi32(max2,max3);
//     return _mm_cvtsi128_si32(max4);
// }


template <int index>
static inline int _mm512_extract_epi32(__m512i target)
{
    return _mm512_cvtsi512_si32(_mm512_alignr_epi32(target, target, index));
}

template<int I = 0>
static inline int mm512_extract_epi32(__m512i target, int index)
{
    if(I == index){
        return _mm512_extract_epi32<I>(target);
    }
    else if constexpr(I < 16){
        return mm512_extract_epi32<I + 1>(target, index);
    }
}

static inline int binarySearch(int32_t *data, int size, int32_t key) {
    int l = 0;
    int h = size - 1;

    while (l <= h) {
		int m = (l + h) / 2;
		
		if (data[m] < key) {
			l = m + 1;
		} else if (data[m] > key) {
			h = m - 1;
		} else {
            while (--m >= 0 && data[m] == key);
			return m;
		}
	}
	return h;
}

static inline int linearSearch(int32_t* data, int32_t size, int32_t low, int32_t high, bool reverse = false){
    if(reverse){
        for (auto i = size - 1; i > -1; i--)
        {
            if (data[i] >= low && data[i] <= high)
            {
                return i;
            }
        }
    }
    else{
        for (auto i = 0; i < size; i++)
        {
            if(data[i] >= low && data[i] <= high){
                return i;
            }
        }
    }

    return size - 1;
}


std::vector<int> mergeSortedArrays(std::vector<std::vector<int>> arrays) 
{
	 std::priority_queue<int, std::vector<int>,std::greater<int> > qt;

	 std::vector<int> res; 
	 auto idx = 0UL;
	 do
	 {
		for(auto i = 0UL ; i < arrays.size(); i++)
	    {
			if(idx < arrays[i].size())
		       qt.push(arrays[i][idx]);
	    }
		res.push_back(qt.top());
		qt.pop();
		idx++;
		     
	 }while(!qt.empty());
	
  return res;
}

template<class T>
static inline void findMinMax(std::vector<T> const& A, T& minValue, T& maxValue){
    std::vector<T> maxValues(omp_get_max_threads(), A[0]);
    std::vector<T> minValues(omp_get_max_threads(), A[0]);

#pragma omp parallel shared(maxValues, minValues)
    {
        auto id = omp_get_thread_num();

        #pragma omp for
        for (auto i = 0UL; i < A.size(); i++){
            if(maxValues[id] < A[i]){
                maxValues[id] = A[i];
            }
            if(minValues[id] > A[i]){
                minValues[id] = A[i];
            }
        }
    }

    minValue = *min_element(minValues.begin(), minValues.end());
    maxValue = *max_element(maxValues.begin(), maxValues.end());

    return;
}

template<class T>
static inline T closestMultiple(T n, int32_t x)
{
	assert(x > 0);

	if (x > n)
		return x - n < n ? x : 0;
 
    n = n + (int)(x/2);
    n = n - (n%x);
    return n ;
}


struct RangeQuery {
	std::vector<int32_t> leftPredicate;
	std::vector<int32_t> rightPredicate;
};

struct TaskClock{
  std::string task = {};
  double duration = {};
     std::chrono::time_point<std::chrono::system_clock> startSystemClock; 
     std::chrono::time_point<std::chrono::system_clock> endSystemClock; 

    TaskClock(){
         duration = 0; 
    }

    TaskClock(std::string taskName): task(taskName){
         task = taskName; 
         duration = 0; 
    }

    inline void start(){
         startSystemClock = std::chrono::system_clock::now(); 
    }

    inline void end(){
         endSystemClock = std::chrono::system_clock::now(); 

         duration += std::chrono::duration<double>(endSystemClock - startSystemClock).count(); 
    }
};

struct TaskHighResClock{
    std::string task;
    long double duration;
    std::chrono::time_point<std::chrono::high_resolution_clock> startHighResClock;
    std::chrono::time_point<std::chrono::high_resolution_clock> endHighResClock;

    TaskHighResClock(){
        duration = 0;
    }
    
    TaskHighResClock(std::string taskName){
        task = taskName;
        duration = 0;
    }

    inline void start(){
        startHighResClock = std::chrono::high_resolution_clock::now();
    }

    inline void end(){
        endHighResClock = std::chrono::high_resolution_clock::now();

        duration += std::chrono::duration_cast<std::chrono::microseconds>(endHighResClock - startHighResClock).count();
    }

};

struct Timer {
    std::unordered_map<std::string, TaskClock> time;
    std::unordered_map<std::string, TaskHighResClock> highResTime;

    inline void start(std::string const& task){
        if(time.find(task) != time.end()){
            time[task].start();
        }
        else{
            time.insert({task, TaskClock(task)});
            time[task].start();
        }

        if(highResTime.find(task) != highResTime.end()){
            highResTime[task].start();
        }
        else{
            highResTime.insert({task, TaskHighResClock(task)});
            highResTime[task].start();
        }
    }

    inline void end(std::string const& task){
        if(time.find(task) != time.end()){
            time[task].end();
        }
        else{
            std::cout << "Job Type Mismatch Error" << std::endl;
            return;
        }

        if(highResTime.find(task) != highResTime.end()){
            highResTime[task].end();
        }
        else{
            std::cout << "Job Type Mismatch Error" << std::endl;
            return;
        }
    }

    inline long double getTaskDuration(std::string task){
        if(time.find(task) != time.end()){
            return time[task].duration;
        }
        else if(task == "index"){
            return  time["sort"].duration + time["merge"].duration + time["partition"].duration + time["distribute"].duration; 
        }
        else if(task == "query"){
            return time["traverse"].duration;
        }
        else{
            return 0.0;
        }
    }

    inline long double getHighResTaskDuration(std::string task){
        if(highResTime.find(task) != highResTime.end()){
            return highResTime[task].duration;
        }
        else if(task == "index"){
            return  highResTime["sort"].duration + highResTime["merge"].duration + highResTime["partition"].duration + highResTime["distribute"].duration; 
        }
        else if(task == "query"){
            return highResTime["traverse"].duration;
        }
        else{
            return 0.0;
        }
    }

    inline long double getCumulativeTime(){
        auto res = 0.0;
        for(auto it = time.begin(); it != time.end(); it++){
            res += (it->second).duration;
        }
        return res;
    }

    inline long double getHighResCumulativeTime(){
        auto res = 0.0;
        for(auto it = highResTime.begin(); it != highResTime.end(); it++){
            res += (it->second).duration;
        }
        return res;
    }
};

struct WorkloadTime{
    int32_t iterations;
    Timer initializeTime;
	std::vector<Timer> times;

	WorkloadTime(int32_t numberOfQueries, int32_t iterations_ = 1): iterations(iterations_) {
		times = std::vector<Timer>(numberOfQueries);
	};

    inline void printQueryTime(int i = 0){
        auto indexTime = times[i].getTaskDuration("index")/iterations;
        if(i == 0){
            indexTime += (initializeTime.getTaskDuration("initialize")/iterations);
        }
        auto queryTime = times[i].getTaskDuration("query")/iterations;
        auto cummulativeTime = indexTime + queryTime;
        auto sortTime = times[i].getTaskDuration("sort")/iterations;
        auto mergeTime = times[i].getTaskDuration("merge")/iterations;
        auto traverseTime = times[i].getTaskDuration("traverse")/iterations;
        auto partitionTime = times[i].getTaskDuration("partition")/iterations;
        auto distributeTime = times[i].getTaskDuration("distribute")/iterations;

        std::cout << indexTime << ";" << queryTime << ";" << times[i].getCumulativeTime() << ";" << cummulativeTime << ";"<< sortTime <<";" << mergeTime << ";" << traverseTime << ";"<< partitionTime << ";"<< distributeTime << std::endl;
    }
};

struct Block{
    public:
        std::unique_ptr<int32_t[]> data;
        int32_t count;
        int32_t sorted;
        int32_t blockSize;

        Block() : count(0), sorted(0), blockSize(BLOCK_SIZE) {
            data = std::unique_ptr<int32_t[]>{
                reinterpret_cast<int32_t *>(malloc(sizeof(int32_t) * blockSize))};
            madvise(data.get(), sizeof(int32_t) * blockSize, MADV_HUGEPAGE);
        }

        Block(int32_t size) : count(0), sorted(0), blockSize(size) {
            data = std::unique_ptr<int32_t[]>{
            reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * blockSize)) };
            madvise(data.get(), sizeof(int32_t) * blockSize, MADV_HUGEPAGE);
        }

        Block(Block&& b) : data(std::move(b.data)), count(b.count), sorted(b.sorted), blockSize(b.blockSize) {}

        Block& operator=(Block&& b){
            if(&b != this){
                data = std::move(b.data);
                count = b.count;
                sorted = b.sorted;
                blockSize = b.blockSize;
            }

            return *this;
        }

        Block(const Block& b) = delete;
        Block &operator=(const Block& b) = delete;

        inline void addElement(int32_t n){
            data[count++] = n;
        }

        inline void bulkAdd(int32_t* nums, int32_t size){
            memcpy(data.get() + count, nums, size * DATA_SIZE_BYTE);
            count += size;
        }

        inline void sortBlock(){
            ska_sort(data.get(), data.get() + count);
            // parallelIntegerSort(data, count, utils::identityF<uintT>());
            sorted = 1;
        }

        inline std::pair<int, int> lookupKeyRange(int32_t low, int32_t high){
            auto lowIndex = sorted ? binarySearch(data.get(), count, low) : linearSearch(data.get(), count, low, high, false);
            auto highIndex = sorted ? binarySearch(data.get(), count, high) : linearSearch(data.get(), count, low, high, true);
            
            return std::make_pair(lowIndex, highIndex);
        }

        void printBlock(){
            std::cout << "---------------" << std::endl;
            for (auto i = 0; i < count; i++)
            {
                std::cout << "|   " << data[i] << "   |" << std::endl;
            }
        }

};

struct Cursor{
    int32_t *current;
    int32_t *end;
};

struct Bucket{
    public:
        Bucket() {
            head.emplace_front(Block());
            cursor.current = head.front().data.get();
            cursor.end = head.front().data.get() + head.front().blockSize;
        }

        Bucket(int32_t size){ 
            head.emplace_front(Block(size));
            cursor.current = head.front().data.get();
            cursor.end = head.front().data.get() + head.front().blockSize;
        }
        
        Bucket(Bucket&& b){
            head = std::move(b.head);
            cursor.current = head.front().data.get();
            cursor.end = head.front().data.get() + head.front().blockSize;
        }

        Bucket& operator=(Bucket&& b){
            if (&b != this)
            {
                head = std::move(b.head);
                cursor.current = head.front().data.get();
                cursor.end = head.front().data.get() + head.front().blockSize;
            }
            
            return *this;
        }

        Bucket(Bucket const & b) = delete;
        Bucket& operator=(Bucket const & b) = delete;

        inline void addElement(int32_t n){
            if(cursor.current >= cursor.end){
                head.emplace_front(Block());
                cursor.current = head.front().data.get();
                cursor.end = head.front().data.get() + head.front().blockSize;
            }
            *(cursor.current) = n;
            cursor.current += 1;
            head.front().count++;
        }

        inline void addElement(int32_t* values, int size = CacheBlockSize){
            if constexpr (CacheBlockSize == 16){
                __m512i castedValues = _mm512_loadu_si512((__m512i*)values);
                if(__builtin_expect(cursor.end - cursor.current < size, 1))
                {
                    head.emplace_front(Block());
                    cursor.current = head.front().data.get();
                    cursor.end = head.front().data.get() + head.front().blockSize;
                }
                _mm512_stream_si512((__m512i *)cursor.current, castedValues);
                cursor.current += size;
                head.front().count += size; /// move 
            }
            else if constexpr(CacheBlockSize == 4){
                __m128i castedValues = _mm_loadu_si128((__m128i*)values);
                if(cursor.end - cursor.current < size)
                {
                    head.emplace_front(Block());
                    cursor.current = head.front().data.get();
                    cursor.end = head.front().data.get() + head.front().blockSize;
                }
                _mm_stream_si128((__m128i *)cursor.current, castedValues);
                cursor.current += size;
                head.front().count += size;
            }
            
        }
        inline void flushElement(int32_t* values, int size){
            if(cursor.end - cursor.current < size)
            {
                head.emplace_front(Block());
                cursor.current = head.front().data.get();
                cursor.end = head.front().data.get() + head.front().blockSize;
            }
            for (auto i = 0; i < size; i++){
                _mm_stream_si32((int32_t*)(cursor.current), (int32_t)values[i]);
                cursor.current++;
            }
            head.front().count += size;
        }

        inline int32_t getSize(){
            return (head.size() - 1) * BLOCK_SIZE + (cursor.end - cursor.current);
        }

        inline void sortBucket(){
            if(head.front().sorted == 1){
                return;
            }
            
            auto newHead = Block(getSize());
            for(auto it = head.begin(); it != head.end(); ++it){
                if((*it).count){
                    newHead.bulkAdd((*it).data.get(), (*it).count);
                }
            }
            head.clear();
            newHead.sortBlock();
            head.emplace_front(std::move(newHead));

            cursor.current = head.front().data.get();
            cursor.end = head.front().data.get() + head.front().blockSize;
        }

        void printBucket(int n){
            std::cout << "+++++++++++++++ BUCKET "<< n << " +++++++++++++++" << std::endl;
            auto i = 0;

            for(auto it = head.begin(); it != head.end(); ++it){
                std::cout << "Block " << i << std::endl;
                (*it).printBlock();
                std::cout << std::endl;
                i++;
            }

            std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
        }

        std::list<Block> head;
        Cursor cursor;
};

struct CacheLineBlock{
    CacheLineBlock() : count(0) {}
    int32_t values[CacheBlockSize];
    int count;
    inline void addElement(int32_t n, Bucket& bucket) {
        values[count] = n;
        ++count;
        if (__builtin_expect(count == CacheBlockSize, 1))
        {
            bucket.addElement(values, CacheBlockSize);
            count = 0;
        }
    }
    inline void flush(Bucket& bucket){
        if(count > 0 ){
            bucket.flushElement(values, count);
            count = 0;
        }
    }
};


Bucket mergeBuckets(Bucket& b1, Bucket& b2){
    Block resBlock = Block(b1.getSize() + b2.getSize());
    Bucket res;
    res.head.emplace_front(std::move(resBlock));
    auto i = 0;
    auto j = 0;
    auto r = 0;

    auto idata = (*b1.head.begin()).data.get();
    auto jdata = (*b2.head.begin()).data.get();
    auto rdata = (*res.head.begin()).data.get();

    while (i < b1.getSize() && j < b2.getSize()){
        if(idata[i] <= jdata[j]){
            rdata[r++] = idata[i++];
            rdata[r++] = jdata[j++];
        }
        else{
            rdata[r++] = jdata[j++];
            rdata[r++] = idata[i++];
        }
    }

    memcpy(rdata + r, idata + i, DATA_SIZE_BYTE * (b1.getSize() - i));
    r += (b1.getSize() - i);
    memcpy(rdata + r, jdata + j, DATA_SIZE_BYTE * (b2.getSize() - j));
    r += (b2.getSize() - j);
    (*res.head.begin()).sorted = 1;

    return res;
}

struct Column {
	int32_t minValue;
	int32_t maxValue;
    std::string distribution;
    Bucket data;

    Column(){}

    Column(int32_t size){
        data = Bucket(size);
    }

    Column(std::vector<int32_t> const& input){
        data = Bucket(input.size());
        
        for (auto n : input)
        {
            data.addElement(n);
        }
        findMinMax(input, minValue, maxValue);
	}

    void loadData(int* input_data, int32_t size){
       memcpy(data.head.front().data.get(), input_data, size * sizeof(int));
    }
};

auto getSample(Column& inputColumn, int32_t dataSize, double samplePercentage = 0.00001){
    int32_t sampleSize = (int)(samplePercentage * dataSize);
    auto samples = std::vector<int>(sampleSize, 0);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,sampleSize);

    for (auto i = 0; i < sampleSize; i++){
        samples[i] = (*(inputColumn.data.head.begin())).data[dist(rng)];
    }

    return samples;
}

template<typename T>
double getMean(std::vector<T>& data){
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

template<typename T>
double getVariance(std::vector<T>& data){
    const double mean = getMean(data);

    const double sum_of_squared_deviations = std::inner_product(
        data.begin(), data.end(),
        data.begin(),
        0.0,
        std::plus<double>(),
        [mean](int x, int y) { return std::pow(x - mean, 2) + std::pow(y - mean, 2); });

    return sum_of_squared_deviations / (data.size() - 1);
}

bool is_normal_distribution(std::vector<int>& data){
    /*
    To determine if a given vector of integers follows a normal distribution, we can use the Kolmogorov-Smirnov test. The steps to perform the test are as follows:

    Sort the vector in ascending order.
    Calculate the mean and standard deviation of the vector.
    For each element in the sorted vector, calculate its corresponding z-score using the formula (element - mean) / standard deviation.
    Calculate the cumulative distribution function (CDF) of the normal distribution at each z-score value using the formula: CDF(z) = 0.5 * (1 + erf(z / sqrt(2))), where erf is the error function.
    Calculate the maximum absolute difference between the empirical CDF of the vector and the normal CDF at each z-score value.
    Calculate the critical value of the Kolmogorov-Smirnov test using the formula: critical_value = 1.36 / sqrt(vector_size).
    If the maximum absolute difference is less than or equal to the critical value, then the data distribution is considered normal and the function returns true. Otherwise, the function returns false.
    */
        // Step 1: sort the vector
    std::vector<int> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    // Step 2: calculate mean and standard deviation
    double mean = getMean(data);
    double variance = getVariance(data);
    double standard_deviation = std::sqrt(variance);

    // Step 3: calculate z-scores
    std::vector<double> z_scores(sorted_data.size());
    for (size_t i = 0; i < sorted_data.size(); ++i) {
        z_scores[i] = (sorted_data[i] - mean) / standard_deviation;
    }

    // Step 4: calculate normal CDF values
    std::vector<double> normal_cdf(sorted_data.size());
    for (size_t i = 0; i < sorted_data.size(); ++i) {
        normal_cdf[i] = 0.5 * (1 + std::erf(z_scores[i] / std::sqrt(2)));
    }

    // Step 5: calculate maximum absolute difference
    double max_diff = 0.0;
    for (size_t i = 0; i < sorted_data.size(); ++i) {
        double diff = std::abs((i + 1.0) / sorted_data.size() - normal_cdf[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    // Step 6: calculate critical value
    double critical_value = 1.36 / std::sqrt(sorted_data.size());

    // Step 7: check if distribution is normal
    return max_diff <= critical_value;
}

bool is_uniform_distribution(std::vector<int>& data) {
    /*
    The function follows a similar approach as the Kolmogorov-Smirnov test for normal distribution, but with some differences:

    The vector is sorted in ascending order.
    The expected value is calculated as the average of the first and last elements of the sorted vector.
    The maximum deviation from the expected value is calculated as the largest absolute difference between any element of the vector and the expected value.
    The critical value is calculated as 1.36 times the range of the vector (i.e., the difference between the largest and smallest values) divided by the square root of the vector size.
    The function returns true if the maximum deviation is less than or equal to the critical value, indicating a uniform distribution, and false otherwise.
    Note that this implementation assumes that the vector is not empty and that all values are positive. If the vector can contain negative values, you may need to adjust the expected value calculation. Additionally, keep in mind that the test may not be suitable for all types of data and that other tests may be more appropriate depending on the specific distribution you are trying to detect.
    */
    // Step 1: sort the vector
    std::vector<int> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    // Step 2: calculate expected value
    double expected_value = (sorted_data.front() + sorted_data.back()) / 2.0;

    // Step 3: calculate maximum deviation from expected value
    double max_deviation = 0.0;
    for (auto value : sorted_data) {
        double deviation = std::abs(value - expected_value);
        if (deviation > max_deviation) {
            max_deviation = deviation;
        }
    }

    // Step 4: calculate critical value
    double critical_value = 1.36 * (sorted_data.back() - sorted_data.front()) / std::sqrt(sorted_data.size());

    // Step 5: check if distribution is uniform
    return max_deviation <= critical_value;
}

bool is_skewed_distribution(std::vector<int>& data) {
    /*
    The function follows a different approach than the Kolmogorov-Smirnov test and the uniform distribution test, as it calculates the skewness of the distribution using the formula:
    skewness = (1/n) * sum((x_i - mean) / std_deviation)^3
    where n is the number of observations, x_i is the value of the i-th observation, mean is the mean of the observations, and std_deviation is the standard deviation of the observations.

    Once the skewness is calculated, the function checks if it is significantly different from zero using a critical value of 1.96 / sqrt(n), which corresponds to a significance level of 5%. If the absolute value of the skewness is greater than this critical value, the function returns true, indicating a skewed distribution, and false otherwise.

    Note that this implementation assumes that the vector is not empty and that all values are positive. If the vector can contain negative values, you may need to adjust the calculation of the mean and standard deviation. Additionally, keep in mind that the test may not be suitable for all types of data and that other tests may be more appropriate depending on the specific distribution you are trying to detect.
    */
    // Step 1: calculate the mean and standard deviation
    double mean = getMean(data);
    double variance = getVariance(data);
    double std_deviation = std::sqrt(variance);

    // Step 2: calculate the skewness
    double skewness = 0.0;
    for (auto value : data) {
        skewness += std::pow((value - mean) / std_deviation, 3);
    }
    skewness /= data.size();

    // Step 3: check if distribution is skewed
    return std::abs(skewness) > 1.96 / std::sqrt(data.size());
}

bool is_spike_distribution(std::vector<int>& data) {
    /*
    The function follows a different approach than the Kolmogorov-Smirnov test, the uniform distribution test, and the skewed distribution test. It uses the quartile range as a measure of the spread of the data, which is defined as the difference between the 75th percentile and the 25th percentile of the sorted data.

    If the quartile range is small compared to the range of the data (i.e., if the quartile range is less than or equal to 25% of the range), the function returns true, indicating a spike distribution, and false otherwise.

    Note that this implementation assumes that the vector is not empty and that all values are positive. If the vector can contain negative values, you may need to adjust the calculation of the quartile range. Additionally, keep in mind that the test may not be suitable for all types of data and that other tests may be more appropriate depending on the specific distribution you are trying to detect.
    */
    // Step 1: sort the vector
    std::vector<int> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    // Step 2: calculate the range of the data
    int range = sorted_data.back() - sorted_data.front();

    // Step 3: calculate the quartile range
    int quartile_range = sorted_data[(3 * sorted_data.size()) / 4] - sorted_data[sorted_data.size() / 4];

    // Step 4: check if the quartile range is small compared to the range
    return quartile_range <= 0.25 * range;
}

bool is_exponential_distribution(std::vector<int>& data) {
    /*
    The function calculates the coefficient of variation (CV) of the data, which is defined as the standard deviation divided by the mean. In an exponential distribution, the CV is always equal to 1. If the CV is smaller than or equal to 1, the function returns true, indicating an exponential distribution, and false otherwise.

    Note that this implementation assumes that the vector is not empty and that all values are positive. If the vector can contain negative values or zeros, you may need to adjust the calculation of the mean and variance. Additionally, keep in mind that the test may not be suitable for all types of data and that other tests may be more appropriate depending on the specific distribution you are trying to detect.
    */
    // Step 1: calculate the mean of the data
    double mean = getMean(data);

    // Step 2: calculate the variance of the data
    double variance = getVariance(data);

    // Step 3: calculate the standard deviation of the data
    double std_deviation = std::sqrt(variance);

    // Step 4: calculate the coefficient of variation
    double cv = std_deviation / mean;

    // Step 5: check if coefficient of variation is small
    return cv <= 1.0;
}

bool is_lognormal_distribution(std::vector<int>& data) {
    /*
    The function applies a logarithmic transformation to the data and then calculates the coefficient of variation (CV) of the transformed data. In a lognormal distribution, the CV of the log-transformed data is always equal to 1. If the CV is smaller than or equal to 1, the function returns true, indicating a lognormal distribution, and false otherwise.

    Note that this implementation assumes that the vector is not empty and that all values are positive. If the vector can contain negative values or zeros, you may need to adjust the transformation of the data. Additionally, keep in mind that the test may not be suitable for all types of data and that other tests may be more appropriate depending on the specific distribution you are trying to detect.
    */
    // Step 1: calculate the mean of the log-transformed data
    std::vector<double> log_data;
    for (auto value : data) {
        log_data.push_back(std::log(value));
    }
    double mean_log_data = std::accumulate(log_data.begin(), log_data.end(), 0.0) / log_data.size();

    // Step 2: calculate the variance of the log-transformed data
    double variance_log_data = getVariance(log_data);

    // Step 3: calculate the standard deviation of the log-transformed data
    double std_deviation_log_data = std::sqrt(variance_log_data);

    // Step 4: calculate the coefficient of variation of the log-transformed data
    double cv_log_data = std_deviation_log_data / mean_log_data;

    // Step 5: check if the coefficient of variation is small
    return cv_log_data <= 1.0;
}

bool is_poisson_distribution(std::vector<int>& data) {
    /*
    The function calculates the chi-squared statistic of the data, which is a commonly used test statistic for testing whether a given set of observations follows a Poisson distribution. The chi-squared statistic is calculated by comparing the observed variance of the data to the expected variance of a Poisson distribution with the same mean. If the chi-squared statistic is less than a critical value at a given level of significance (e.g., 95%), the null hypothesis that the data follows a Poisson distribution cannot be rejected.

    Note that this implementation assumes that the vector is not empty and that all values are non-negative integers. If the vector can contain negative values or non-integer values, you may need to adjust the calculation of the expected variance and the degrees of freedom. Additionally, keep in mind that the test may not be suitable for all types of data and that other tests may be more appropriate depending on the specific distribution you are trying to detect.
    */
    // Step 1: calculate the mean of the data
    double mean = getMean(data);

    // Step 2: calculate the variance of the data
    double variance = getVariance(data);

    // Step 3: calculate the expected variance of a Poisson distribution with the same mean
    double expected_variance = mean;

    // Step 4: calculate the chi-squared statistic
    double chi_squared = variance / expected_variance;

    // Step 5: calculate the degrees of freedom
    // int degrees_of_freedom = data.size() - 1;

    // Step 6: calculate the critical value at the 95% confidence level
    double critical_value = 3.84;

    // Step 7: check if the chi-squared statistic is less than the critical value
    return chi_squared < critical_value;
}

bool is_gamma_distribution(std::vector<int>& data) {
    /*
    The function calculates the chi-squared statistic of the data, which is a commonly used test statistic for testing whether a given set of observations follows a gamma distribution. The chi-squared statistic is calculated by comparing the observed frequencies of the data to the expected frequencies of a gamma distribution with the same shape and scale parameters. If the chi-squared statistic is less than a critical value at a given level of significance (e.g., 95%), the null hypothesis that the data follows a gamma distribution cannot be rejected.

    Note that this implementation assumes that the vector is not empty and that all values are non-negative integers. If the vector can contain negative values or non-integer values, you may need to adjust the calculation of the expected values. Additionally, keep in mind that the test may not be suitable for all types of data and that other tests may be more appropriate depending on the specific distribution you are trying to detect.
    */
    // Step 1: calculate the mean and variance of the data
    double mean = getMean(data);
    double variance = getVariance(data);

    // Step 2: calculate the shape and scale parameters of the gamma distribution
    double shape = std::pow(mean, 2) / variance;
    double scale = variance / mean;

    // Step 3: calculate the chi-squared statistic
    double sum_squared_deviation = 0.0;
    for (auto value : data) {
        double expected_value = std::tgamma(shape + 1.0) * std::pow(scale, shape) * std::pow(value, shape - 1) * std::exp(-value * scale);
        sum_squared_deviation += std::pow(value - expected_value, 2) / expected_value;
    }
    double chi_squared = sum_squared_deviation / data.size();

    // Step 4: calculate the degrees of freedom
    // int degrees_of_freedom = data.size() - 2;

    // Step 5: calculate the critical value at the 95% confidence level
    double critical_value = 5.99;

    // Step 6: check if the chi-squared statistic is less than the critical value
    return chi_squared < critical_value;
}

bool is_weibull_distribution(std::vector<int>& data) {
    /*
    The function calculates the Kolmogorov-Smirnov statistic of the data, which is a commonly used test statistic for testing whether a given set of observations follows a Weibull distribution. The Kolmogorov-Smirnov statistic is calculated by comparing the cumulative distribution function (CDF) of the data to the CDF of a Weibull distribution with the same shape and scale parameters. If the maximum deviation between the two CDFs is less than a critical value at a given level of significance (e.g., 95%), the null hypothesis that the data follows a Weibull distribution cannot be rejected.

    Note that this implementation assumes that the vector is not empty and that all values are non-negative integers. If the vector can contain negative values or non-integer values, you may need to adjust the calculation of the expected values. Additionally, keep in mind that the test may not be suitable for all types of data and that other tests may be more appropriate depending on the specific distribution you are trying to detect.
    */
    // Step 1: calculate the mean and variance of the data
    double mean = getMean(data);
    double variance = getVariance(data);

    // Step 2: calculate the shape and scale parameters of the Weibull distribution
    double shape = 1.0;
    double scale = std::sqrt(variance / std::tgamma(1.0 + 2.0 / shape)) / std::pow(mean, 1.0 / shape);

    // Step 3: calculate the Kolmogorov-Smirnov statistic
    std::sort(data.begin(), data.end());
    double max_deviation = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        double cdf = 1.0 - std::exp(-std::pow(data[i] / scale, shape));
        double deviation = std::fabs(cdf - static_cast<double>(i + 1) / data.size());
        if (deviation > max_deviation) {
            max_deviation = deviation;
        }
    }

    // Step 4: calculate the critical value at the 95% confidence level
    double critical_value = 1.36 / std::sqrt(data.size());

    // Step 5: check if the Kolmogorov-Smirnov statistic is less than the critical value
    return max_deviation < critical_value;
}

auto getDistribution(std::vector<int>& data){
    if(is_normal_distribution(data)){
        return "Normal Distribution";
    }
    else if(is_uniform_distribution(data)){
        return "Uniform Distribution";
    }
    else if(is_spike_distribution(data)){
        return "Spike Distribution";
    }
    else if(is_gamma_distribution(data)){
        return "Gamma distribution";
    }
    else if(is_weibull_distribution(data)){
        return "Weibull distribution";
    }
    else if(is_skewed_distribution(data)){
        return "Skewed distribution";
    }
    else if(is_poisson_distribution(data)){
        return "Poisson distribution";
    }
    else if(is_exponential_distribution(data)){
        return "Exponential distribution";
    }
    else if(is_lognormal_distribution(data)){
        return "Lognormal distribution";
    }

    return "Other Distribution"; 
}

auto loadColumn(Column& inputColumn, std::string COLUMN_FILE_PATH, int32_t COLUMN_SIZE) {
	FILE *f = fopen(COLUMN_FILE_PATH.c_str(), "r");
	if (!f) {
		printf("Cannot open file.\n");
		return 0UL;
	}
	int *temp_data = (int *)malloc(sizeof(int) * COLUMN_SIZE);
	auto res = fread(temp_data, sizeof(int), COLUMN_SIZE, f);
	inputColumn.minValue = std::numeric_limits<int32_t>::max();
	inputColumn.maxValue = std::numeric_limits<int32_t>::min();
    for (auto i = 0; i < COLUMN_SIZE; i++)
    {
        inputColumn.data.addElement(temp_data[i]);

		if (temp_data[i] < inputColumn.minValue) {
			inputColumn.minValue = temp_data[i];
		}
		if (temp_data[i] > inputColumn.maxValue) {
			inputColumn.maxValue = temp_data[i];
		}
    }
    auto samples = getSample(inputColumn, COLUMN_SIZE);
    inputColumn.distribution = getDistribution(samples);

    free(temp_data);
	fclose(f);

    return res;
}

auto loadQueries(RangeQuery& queries, std::string QUERIES_FILE_PATH, int32_t NUM_QUERIES) {
	FILE *f = fopen(QUERIES_FILE_PATH.c_str(), "r");
	if (!f) {
		printf("Cannot open file.\n");
		return 0UL;
	}
	int64_t *temp_data = (int64_t *)malloc(sizeof(int64_t) * NUM_QUERIES);
	auto res = fread(temp_data, sizeof(int64_t), NUM_QUERIES, f);
	queries.leftPredicate = std::vector<int32_t>(NUM_QUERIES);
	for (auto i = 0; i < NUM_QUERIES; i++) {
		queries.leftPredicate[i] = temp_data[i];
	}
	res = fread(temp_data, sizeof(int64_t), NUM_QUERIES, f);
	queries.rightPredicate = std::vector<int32_t>(NUM_QUERIES);
	for (auto i = 0; i < NUM_QUERIES; i++) {
		queries.rightPredicate[i] = temp_data[i];
	}
	free(temp_data);
	fclose(f);

    return res;
}


#endif
