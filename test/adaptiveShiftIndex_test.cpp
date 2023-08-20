#ifndef __DERAPA_TEST__
#define __DERAPA_TEST__

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "constants.h"
#include "../header/adaptiveShiftIndex.hpp"


vector<int32_t> generateData(/*string filePath = ""*/){
    vector<int32_t> data;
    for (auto i = 0; i < 100; i += 3)
    {
        data.emplace_back((i+8) % 70);
    }
    
    return data;
}


TEST_CASE("First and single Bucket","[first and single bucket]"){
    Timer timer;
    static auto domain = __itt_domain_create("DERAPA-Tests");
    __itt_resume();
    Column c(generateData());
    AdaptiveShiftIndex idx(&c, timer, domain);
    idx.rangeQuery(1, 8, timer, domain);
    auto res = idx.countResults(1, 8, timer, domain);
    __itt_pause();
    REQUIRE(res == 3);
}

TEST_CASE("Last and multiple Buckets","[last and multiple buckets]"){
    Timer timer;
    static auto domain = __itt_domain_create("DERAPA-Tests");
    __itt_resume();
    Column c(generateData());
    AdaptiveShiftIndex idx(&c, timer, domain);
    idx.rangeQuery(10, 18, timer, domain);
    auto res = idx.countResults(10, 18, timer, domain);
    __itt_pause();
    REQUIRE(res == 6);
}

TEST_CASE("Empty Bucket","[empty bucket]"){
    Timer timer;
    static auto domain = __itt_domain_create("DERAPA-Tests");
    __itt_resume();
    Column c(generateData());
    AdaptiveShiftIndex idx(&c, timer, domain);
    idx.rangeQuery(90, 98, timer, domain);
    auto res = idx.countResults(90, 98, timer, domain);
    __itt_pause();
    REQUIRE(res == 0);
}

TEST_CASE("Edge Bucket","[edge bucket]"){
    Timer timer;
    static auto domain = __itt_domain_create("DERAPA-Tests");
    __itt_resume();
    Column c(generateData());
    AdaptiveShiftIndex idx(&c, timer, domain);
    idx.rangeQuery(50, 65, timer, domain);
    auto res = idx.countResults(50, 65, timer, domain);
    __itt_pause();
    REQUIRE(res == 5);
}

#endif