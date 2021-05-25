#pragma once
#include "Algorithm.h"

using namespace std;

class RandomizeAlg : public Algorithm
{
public:
    RandomizeAlg(const Dataset& d, const vector<uint>& budgets);
    double get_solutions(vector<bool> &seeds);
};