#pragma once
#include "Algorithm.h"

using namespace std;

class Soda : public Algorithm
{
public:
    Soda(const Dataset &d, const vector<uint> &budgets);
    double get_solutions(vector<bool> &seeds);

private:
    void split(vector<bool> &seed_A, vector<bool> &seed_B,
               vector<bool> &seed_A_B,
               vector<uint> &select_count_A, vector<uint> &select_count_B,
               vector<uint> &select_count_A_B,
               double &obj_A, double &obj_B,
               set<uint> &I_A, set<uint> &I_B,
               vector<set<uint>> &groups_A, vector<set<uint>> &groups_B,
               set<uint> I, vector<set<uint>> groups);

    void residual_greedy(vector<bool> &seeds, double &obj,
                         vector<uint> &select_count,
                         set<uint> &I, vector<set<uint>> &groups);

    double p;
};