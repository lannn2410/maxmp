#include "Constants.h"

const int Constants::NUM_THREADS = 70;
const int Constants::NUM_SAMPLES = 100;
APP_TYPE Constants::APPLICATION = APP_TYPE::Social;
int Constants::NUM_GROUPS = 2;
int Constants::TOTAL_BUDGET = 10;
const int Constants::NUM_REPEAT = 5;

const int Constants::PROB_NO_BOOST = 1;
const int Constants::PROB_BOOST = 2;
int Constants::NUM_INFLUENCE = 1;

const KERNEL_TYPE Constants::KERNEL = KERNEL_TYPE::GAUSSIAN;
const double Constants::KERNEL_GAMMA = 1e-6;
bool Constants::LOG_OBJ = false;

const double Constants::RANDOMIZE_DELTA = 0.01;
double Constants::FAST_THRESHOLD_EPSILON = 0.5;
const bool Constants::FAST_THRESHOLD_SOFT_GREEDY = false;
const double Constants::SODA_BETA = 0.4;
