#ifndef __DERAPA__
#define __DERAPA__

#include "constants.h"
#include "../../header/adaptiveShiftIndex.hpp"
#include <math.h>

#endif

pair<string, string> split_once(string delimited, char delimiter) {
	auto pos = delimited.find_first_of(delimiter);
	return {delimited.substr(0, pos), delimited.substr(pos + 1)};
}

void print_help(int argc, char **argv) {
	cerr<<"Unrecognized command line option."<<endl;
	cerr<<"Usage: "<< argv[0]<<" [args]"<<endl;
	cerr<<"   --column-path"<<endl;
	cerr<<"   --query-path"<<endl;
	cerr<<"   --answers-path"<<endl;
	cerr<<"   --num-queries"<<endl;
	cerr<<"   --column-size"<<endl;
}

/// allocate Blocks
void initializePagePool(int n){
	vector<Block *> ptrs(n);
	for (auto i = 0; i < n; i++){
		auto tmp = new Block();
		tmp->addElement(rand() + 1);
		ptrs.emplace_back(tmp);
	}

	for (auto i = 0; i < n; i++){
		free(ptrs[i]);
	}
}

int main(int argc, char** argv)
{
    auto iterations = 1;

    auto NUM_QUERIES = 158325;
    auto COLUMN_SIZE = 585624220;
    string COLUMN_FILE_PATH = "/home/ak10318/data/skyserver/skyserver.data";
    string QUERIES_FILE_PATH = "/home/ak10318/data/skyserver/query_0.1";
    string ANSWER_FILE_PATH = "/home/ak10318/data/skyserver/answer_0.1";

	// auto NUM_QUERIES = 10000;
    // auto COLUMN_SIZE = 100000000;
	// string COLUMN_FILE_PATH = "/home/ak10318/data/synthetic/skew_col_100M";
    // string QUERIES_FILE_PATH = "/home/ak10318/data/synthetic/skew_col_seqOver_query_10K";
    // string ANSWER_FILE_PATH = "/home/ak10318/data/synthetic/skew_col_seqOver_query_answers_10K";

    RangeQuery rangeQueries;
    

	for (int i = 1; i < argc; i++) {
		auto arg = string(argv[i]);
		if (arg.substr(0, 2) != "--") {
			print_help(argc, argv);
			exit(EXIT_FAILURE);
		}
		arg = arg.substr(2);
		auto p = split_once(arg, '=');
		auto &arg_name = p.first;
		auto &arg_value = p.second;
		if (arg_name == "column-path") {
			COLUMN_FILE_PATH = arg_value;
		} else if (arg_name == "query-path") {
			QUERIES_FILE_PATH = arg_value;
		} else if (arg_name == "answer-path") {
			ANSWER_FILE_PATH = arg_value;
		} else if (arg_name == "num-queries") {
			NUM_QUERIES = atoi(arg_value.c_str());
		} else if (arg_name == "column-size") {
			COLUMN_SIZE = atoi(arg_value.c_str());
		} else {
			print_help(argc, argv);
			exit(EXIT_FAILURE);
		}
	}

    
	WorkloadTime wt(NUM_QUERIES, iterations);
    loadQueries(rangeQueries, QUERIES_FILE_PATH, NUM_QUERIES);

	// initializePagePool(ceil(COLUMN_SIZE / BLOCK_SIZE));
	// Column c(COLUMN_SIZE);
	// loadColumn(c, COLUMN_FILE_PATH, COLUMN_SIZE);

	for (auto j = 0; j < iterations; j++)
	{
		initializePagePool(ceil(COLUMN_SIZE / BLOCK_SIZE));
		Column c(COLUMN_SIZE);
		loadColumn(c, COLUMN_FILE_PATH, COLUMN_SIZE);

		static auto domain = __itt_domain_create("DERAPA");
		__itt_resume();

		auto index = AdaptiveShiftIndex(&c, wt.initializeTime, domain);

	    for (auto i = 0; i < 1/*NUM_QUERIES*/; i++)
	    {
	        auto res = index.rangeQuery(rangeQueries.leftPredicate[i], rangeQueries.rightPredicate[i],wt.times.at(i),  domain);
		}

		__itt_pause();
	}

	for (auto i = 0; i < 1/*NUM_QUERIES*/; i++)
	{
	    wt.printQueryTime(i);
	}

	return 0;
}
