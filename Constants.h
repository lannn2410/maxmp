#pragma once
#include <string>

using namespace std;

enum class APP_TYPE {Social, Video};
enum class KERNEL_TYPE {GAUSSIAN, VECTOR_PRODUCT};

class Constants
{
public:
	static const int NUM_THREADS;
	static const int NUM_SAMPLES;
    static APP_TYPE APPLICATION;
    static int NUM_GROUPS;
    static int TOTAL_BUDGET;
    
    static const int NUM_REPEAT; // no. repeat run of a randomize alg
    
    // for social network application
    static const int PROB_NO_BOOST;
    static const int PROB_BOOST;
    static int NUM_INFLUENCE; // no. node to start influence

    // for video summarization
    static const KERNEL_TYPE KERNEL;
    static const double KERNEL_GAMMA;
    static bool LOG_OBJ; // convert obj to log(obj)

    // for algorithm
    static const double RANDOMIZE_DELTA;
    static double FAST_THRESHOLD_EPSILON;
    static const bool FAST_THRESHOLD_SOFT_GREEDY; // compute max gain in each loop
    static const double SODA_BETA;
};

