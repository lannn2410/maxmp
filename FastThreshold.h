#pragma once
#include "Algorithm.h"

using namespace std;

class FastThreshold : public Algorithm
{
public:
    FastThreshold(const Dataset& d, const vector<uint>& budgets);
    double get_solutions(vector<bool> &seeds);
};