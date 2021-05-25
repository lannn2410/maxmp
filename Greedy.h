#pragma once
#include "Algorithm.h"

using namespace std;

class Greedy : public Algorithm
{
public:
    Greedy(const Dataset& d, const vector<uint>& budgets);
    double get_solutions(vector<bool> &seeds);
};