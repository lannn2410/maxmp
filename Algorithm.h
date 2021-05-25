#pragma once
#include "Dataset.h"
#include "Constants.h"
#include "MpCommon.h"

using namespace std;

class Algorithm
{
public:
    Algorithm(const Dataset &d, const vector<uint> &budgets);
    virtual double get_solutions(vector<bool> &seeds) = 0; // seeds returned in form of 0-1 vector
    uint get_num_queries() const;
    void reset_num_queries();

protected:
    void initiate(vector<bool> &seeds,
                  vector<set<uint>> &groups, vector<uint> &select_count,
                  double &B, set<uint> &I, double &initial_obj);
    double query(const vector<bool> &s);
    const Dataset &data;
    const vector<uint> &budgets;
    MpCommon *common_ins;
    uint num_queries;
};
