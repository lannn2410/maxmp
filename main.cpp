#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip> // setprecision
#include <string>
#include <sstream>
#include <numeric>
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#define PORT 8080

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "Constants.h"
#include "MpCommon.h"
#include "Dataset.h"
#include "RandomizeAlg.h"
#include "FastThreshold.h"
#include "Greedy.h"
#include "RandomGreedy.h"
#include "Soda.h"

using namespace std;

enum CHANGE
{
	BUDGET = 0,
	NUM_GROUPS = 1
};
enum ALG
{
	RANDOMIZE = 0,
	FAST_2 = 1,
	FAST_5 = 2,
	FAST_9 = 3,
	GREEDY = 4,
	RANDOM_GREEDY = 5,
	SODA = 6
};

static const char *AppString[] = {"SOCIAL", "VIDEO"};
static const char *AlgString[] = {"RANDOMIZE", "FAST_2", "FAST_5", "FAST_9", "GREEDY", "RANDOM_GREEDY", "SODA"};
static const char *ChangeString[] = {"BUDGET", "NUM_GROUP"};

struct Result
{
	// can run many times for one algorithms
	ALG algorithm;
	vector<double> obj{};
	vector<uint> num_queries{};
	vector<uint64_t> runtime{};
	vector<vector<bool>> selected_ids;
};

uint64_t timeMillisec()
{
	using namespace std::chrono;
	return duration_cast<milliseconds>(
			   system_clock::now().time_since_epoch())
		.count();
}

Result
getResultAlgorithm(const Dataset &d, const ALG &alg_kind,
				   const vector<uint> &budgets,
				   uint num_repeat = 1 /*number of repeats*/)
{
	Result re;
	Algorithm *alg;

	if (true)
	{
		switch (alg_kind)
		{
		case ALG::RANDOMIZE:
			alg = new RandomizeAlg(d, budgets);
			break;
		case ALG::FAST_2:
		{
			Constants::FAST_THRESHOLD_EPSILON = 0.2;
			alg = new FastThreshold(d, budgets);
			break;
		}
		case ALG::FAST_5:
		{
			Constants::FAST_THRESHOLD_EPSILON = 0.5;
			alg = new FastThreshold(d, budgets);
			break;
		}
		case ALG::FAST_9:
		{
			Constants::FAST_THRESHOLD_EPSILON = 0.9;
			alg = new FastThreshold(d, budgets);
			break;
		}
		case ALG::GREEDY:
			alg = new Greedy(d, budgets);
			break;
		case ALG::RANDOM_GREEDY:
			alg = new RandomGreedy(d, budgets);
			break;
		case ALG::SODA:
			alg = new Soda(d, budgets);
			break;
		default:
			alg = new RandomizeAlg(d, budgets);
			break;
		}
	}

	for (int i = 0; i < num_repeat; ++i)
	{
		vector<bool> seeds(d.get_data_size(), false);
		alg->reset_num_queries();

		// run the algorithm
		auto start = timeMillisec();
		auto obj = alg->get_solutions(seeds);
		if (Constants::APPLICATION == APP_TYPE::Video && Constants::LOG_OBJ)
		{
			obj = exp(obj);
		}
		auto num_queries = alg->get_num_queries();
		auto dur = timeMillisec() - start;

		spdlog::info("Done {} for {} times, obj {}, no. queries {}, runtime {}",
					 AlgString[(int)alg_kind], i, obj, num_queries, dur);

		// save to result
		re.algorithm = alg_kind;
		re.obj.emplace_back(obj);
		re.num_queries.emplace_back(num_queries);
		re.selected_ids.emplace_back(seeds);
		re.runtime.emplace_back(dur);
	}

	delete alg;
	return re;
}

void make_folder(const string &folder_path)
{
#if defined(_WIN32)
	mkdir(folder_path.c_str());
#else
	mkdir(folder_path.c_str(), 0777); // notice that 777 is different than 0777
#endif
}

vector<Result> runAlgorithms(const Dataset &data)
{
	spdlog::info("Total budget {}, no. groups {}",
				 Constants::TOTAL_BUDGET, Constants::NUM_GROUPS);
	vector<Result> re;
	uint budget_per_group = Constants::TOTAL_BUDGET / Constants::NUM_GROUPS;
	uint redundance = Constants::TOTAL_BUDGET % Constants::NUM_GROUPS;
	vector<uint> budgets(Constants::NUM_GROUPS, budget_per_group);
	for (int i = 0; i < redundance; ++i)
	{
		++budgets[i];
	}
	// budgets[Constants::NUM_GROUPS - 1] = Constants::TOTAL_BUDGET -
	// 									 budget_per_group * (Constants::NUM_GROUPS - 1);
	for (int i = (int)ALG::RANDOMIZE; i <= (int)ALG::SODA; ++i)
	{
		ALG alg = static_cast<ALG>(i);
		if (alg == ALG::RANDOMIZE || alg == ALG::RANDOM_GREEDY)
		{
			re.emplace_back(
				getResultAlgorithm(data, alg, budgets, Constants::NUM_REPEAT));
		}
		else
		{
			re.emplace_back(
				getResultAlgorithm(data, alg, budgets));
		}
	}
	return re;
}

void runExperiment(const APP_TYPE &application, const string &file_name,
				   const uint &num_nodes = 0 /*for social app only*/,
				   const uint &W = 1920, const uint &H = 798, // for video app
				   const CHANGE &change = CHANGE::BUDGET,
				   const uint &min = 1, const uint &max = 10, const uint &step = 1)
{
	Constants::APPLICATION = application;
	long folder_prefix = time(NULL);

	const string re_folder = "result/" + to_string(folder_prefix) +
							 "_" + file_name + "_" + AppString[(int)application] +
							 "_" + ChangeString[change];

	spdlog::info("Run experiments with file {}, change {}",
				 file_name, ChangeString[change]);
	spdlog::info("Results will be stored in {}", re_folder);

	make_folder(re_folder);

	ofstream writefile_query(re_folder + "/query.csv");
	ofstream writefile_sol(re_folder + "/solution.csv");
	ofstream writefile_runtime(re_folder + "/runtime.csv");

	if (!writefile_query.is_open() || !writefile_sol.is_open() ||
		!writefile_runtime.is_open())
		throw "files cannot be written";

	const string header = "B,num_groups,Randomize,Fast_2,Fast_5,Fast_9,Greedy,Random_Greedy,Soda";

	writefile_query << header << endl;
	writefile_sol << header << endl;
	writefile_runtime << header << endl;

	// lambda function to write results to files
	auto writeResultToFile = [&writefile_query, &writefile_sol, &writefile_runtime,
							  &re_folder, &application, &file_name](vector<Result> const &results) {
		writefile_query << Constants::TOTAL_BUDGET << ","
						<< Constants::NUM_GROUPS << ",";
		writefile_sol << Constants::TOTAL_BUDGET << ","
					  << Constants::NUM_GROUPS << ",";
		writefile_runtime << Constants::TOTAL_BUDGET << ","
						  << Constants::NUM_GROUPS << ",";

		for (auto const &result : results)
		{
			if (result.obj.empty())
				throw "Error: No result";
			double obj = accumulate(result.obj.begin(), result.obj.end(), 0);
			obj /= result.obj.size();
			double query = accumulate(result.num_queries.begin(),
									  result.num_queries.end(), 0);
			query /= result.num_queries.size();
			double runtime = accumulate(result.runtime.begin(),
										result.runtime.end(), 0);
			runtime /= result.runtime.size();

			writefile_sol << obj << ",";
			writefile_query << query << ",";
			writefile_runtime << runtime << ",";
		}
		writefile_query << endl;
		writefile_sol << endl;
		writefile_runtime << endl;

		if (application == APP_TYPE::Social)
			return;

		// copy images if video application
		const string image_folder = re_folder + "/B_" + to_string(Constants::TOTAL_BUDGET) +
									"_G_" + to_string(Constants::NUM_GROUPS);
		make_folder(image_folder);
		for (auto const &result : results)
		{
			if (result.obj.empty())
				throw "Error: No result";
			const int alg_idx = result.algorithm;
			const string alg_folder = image_folder + "/" + AlgString[alg_idx];
			make_folder(alg_folder);
			uint c = 1;
			for (auto const &sel_ids : result.selected_ids)
			{
				const string attempt_folder = alg_folder + "/attempt_" + to_string(c);
				make_folder(attempt_folder);
				// copy images to folder
				for (int i = 0; i < sel_ids.size(); ++i)
				{
					if (sel_ids[i])
					{
						string cmd = "cp data/frames/" + file_name + "/" +
									 to_string(i + 1) + ".jpg " + attempt_folder;
						system(cmd.c_str());
					}
				}
				++c;
			}
		}
	};

	Dataset data;
	if (application == APP_TYPE::Social)
	{
		data.read_network(num_nodes, "data/" + file_name, Constants::NUM_INFLUENCE);
	}
	else
	{
		data.read_video("data/" + file_name, W, H);
	}

	if (change == CHANGE::BUDGET)
	{
		data.form_groups(Constants::NUM_GROUPS);
		for (Constants::TOTAL_BUDGET = min;
			 Constants::TOTAL_BUDGET <= max;
			 Constants::TOTAL_BUDGET += step)
		{

			writeResultToFile(runAlgorithms(data));
		}
	}
	else
	{
		for (Constants::NUM_GROUPS = min;
			 Constants::NUM_GROUPS <= max;
			 Constants::NUM_GROUPS += step)
		{
			data.form_groups(Constants::NUM_GROUPS);
			writeResultToFile(runAlgorithms(data));
		}
	}
	writefile_query.close();
	writefile_sol.close();
	writefile_runtime.close();
}

void print_help()
{
	cout << "Options: " << endl;
	cout << "-h <print help>" << endl
		 << "-t <application type, 0: social, 1: video> # default: 0" << endl
		 << "-a <algorithm type, 0: drand, 1: fast, 2: greedy, 3: res.greedy, 4: splitgrow> # default: 0" << endl
		 << "-b <total budget> # default: 20" << endl
		 << "-k <number of groups> # default: 2" << endl
		 << "-p <number of threads> # default: 10" << endl;
}

ALG getAlgType(int a)
{
	switch (a)
	{
	case 0:
		return ALG::RANDOMIZE;
		break;
	case 1:
		return ALG::FAST_5;
		break;
	case 2:
		return ALG::GREEDY;
		break;
	case 3:
		return ALG::RANDOM_GREEDY;
		break;
	case 4:
		return ALG::SODA;
		break;
	default:
		return ALG::RANDOMIZE;
		break;
	}
}

struct Args
{
	APP_TYPE type = APP_TYPE::Social;
	ALG alg = ALG::RANDOMIZE;
	int b = 20, k = 2, p = 10; // total budget, no. group - budget per group = b / k
	bool is_help = false;
};

Args parseArgs(int argc, char **argv)
{
	Args re;
	int i = 1;
	while (i <= argc - 1)
	{
		string arg = argv[i];
		if (arg == "-h")
		{
			re.is_help = true;
			break;
		}
		if (i + 1 >= argc)
			break;
		string s_val = argv[i + 1];
		if (s_val == "-h")
		{
			re.is_help = true;
			break;
		}
		std::string::size_type sz;
		if (arg == "-t" || arg == "-a" || arg == "-b" ||
			arg == "-k" || arg == "-p")
		{
			int val = std::stoi(s_val, &sz);
			if (arg == "-t")
			{
				if (val < 0 || val > 1)
				{
					cout << "Application is not valid" << endl;
					re.is_help = true;
					break;
				}
				re.type = val == 0 ? APP_TYPE::Social : APP_TYPE::Video;
			}
			else if (arg == "-a")
			{
				if (val < 0 || val > 4)
				{
					cout << "Alg is not valid" << endl;
					re.is_help = true;
					break;
				}
				re.alg = getAlgType(val);
			}
			else if (arg == "-b")
			{
				if (val < 1)
				{
					cout << "b < 1" << endl;
					re.is_help = true;
					break;
				}
				re.b = val;
			}
			else if (arg == "-k")
			{
				if (val < 1)
				{
					cout << "k < 1" << endl;
					re.is_help = true;
					break;
				}
				re.k = val;
			}
			else if (arg == "-p")
			{
				if (val < 1)
				{
					cout << "Num. threads is not valid" << endl;
					re.is_help = true;
					break;
				}
				re.p = val;
			}
		}
		else
		{
			cout << "Wrong arguments" << endl;
			re.is_help = true;
			break;
		}

		i += 2;
	}

	return re;
}

void run_command(const Args &r)
{
	if (r.is_help)
	{
		print_help();
		return;
	}
	Constants::APPLICATION = r.type;
	Constants::TOTAL_BUDGET = r.b;
	Constants::NUM_GROUPS = r.k;
	omp_set_num_threads(r.p);
	Dataset data;
	if (r.type == APP_TYPE::Social)
	{
		data.read_network(4039, "data/facebook_combined.txt", Constants::NUM_INFLUENCE); // hard code
	}
	else
	{
		data.read_video("data/cooking.mp4", 1920, 1080);
	}
	data.form_groups(Constants::NUM_GROUPS);
	// form budget per group
	uint budget_per_group = Constants::TOTAL_BUDGET / Constants::NUM_GROUPS;
	uint redundance = Constants::TOTAL_BUDGET % Constants::NUM_GROUPS;
	vector<uint> budgets(Constants::NUM_GROUPS, budget_per_group);
	for (int i = 0; i < redundance; ++i)
	{
		++budgets[i];
	}

	Result res = getResultAlgorithm(data, r.alg, budgets);

	if (res.obj.empty())
	{
		cout << "The algorithm ran unsucessfully" << endl;
		return;
	}
	cout << "Objective: " << res.obj[0] << endl
		 << "Number of queries: " << res.num_queries[0] << endl;
		//  << "Runtime: " << res.runtime[0] << endl;
	
	// if (r.type == APP_TYPE::Video) {
	// 	cout << "Selected frames: ";
	// 	for (auto i=0; i<res.selected_ids[0].size(); ++i)
	// 	{
	// 		if (res.selected_ids[0][i])
	// 			cout << i + 1 << " ";
	// 	}
	// 	cout << endl;
	// }
}

int main(int argc, char *argv[])
{
	srand(time(NULL));
	run_command(parseArgs(argc, argv));
	// omp_set_num_threads(Constants::NUM_THREADS);

	// Constants::FAST_THRESHOLD_EPSILON = 0.5;
	// Constants::NUM_GROUPS = 2;
	// runExperiment(APP_TYPE::Video, "cooking.mp4",
	// 			  0, 1920, 1080, CHANGE::BUDGET, 10, 20, 1);

	// Constants::TOTAL_BUDGET = 20;
	// runExperiment(APP_TYPE::Video, "cooking.mp4",
	// 			  0, 1920, 1080, CHANGE::NUM_GROUPS, 2, 20, 2);

	// Constants::NUM_GROUPS = 2;
	// runExperiment(APP_TYPE::Video, "avenger_trailer.mp4",
	// 			  0, 1920, 798, CHANGE::BUDGET, 10, 20, 1);

	// Constants::TOTAL_BUDGET = 20;
	// runExperiment(APP_TYPE::Video, "avenger_trailer.mp4",
	// 			  0, 1920, 798, CHANGE::NUM_GROUPS, 2, 20, 2);

	// // Constants::FAST_THRESHOLD_EPSILON = 0.9;
	// Constants::NUM_GROUPS = 2;
	// runExperiment(APP_TYPE::Social, "facebook_combined.txt",
	// 			  4039, 0, 0, CHANGE::BUDGET, 10, 100, 10);

	// Constants::TOTAL_BUDGET = 100;
	// runExperiment(APP_TYPE::Social, "facebook_combined.txt",
	// 			  4039, 0, 0, CHANGE::NUM_GROUPS, 2, 20, 1);

	// Constants::FAST_THRESHOLD_EPSILON = 0.5;
	// Constants::NUM_GROUPS = 2;
	// runExperiment(APP_TYPE::Video, "racing.mp4",
	// 			  0, 640, 360, CHANGE::BUDGET, 10, 20, 1);

	// Constants::TOTAL_BUDGET = 18;
	// runExperiment(APP_TYPE::Video, "racing.mp4",
	// 			  0, 640, 360, CHANGE::NUM_GROUPS, 2, 18, 2);

	// Constants::NUM_GROUPS = 2;
	// runExperiment(APP_TYPE::Video, "tom_jerry.mp4",
	// 			  0, 640, 360, CHANGE::BUDGET, 10, 30, 2);

	// Constants::TOTAL_BUDGET = 30;
	// runExperiment(APP_TYPE::Video, "tom_jerry.mp4",
	// 			  0, 640, 360, CHANGE::NUM_GROUPS, 2, 30, 2);

	return 1;
}