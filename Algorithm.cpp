#include "Algorithm.h"

Algorithm::Algorithm(const Dataset &d, const vector<uint> &budgets)
    : data(d), budgets(budgets), num_queries(0)
{
    common_ins = MpCommon::getInstance();
}

uint Algorithm::get_num_queries() const
{
    return num_queries;
}

void Algorithm::initiate(vector<bool> &seeds,
                         vector<set<uint>> &groups, vector<uint> &select_count,
                         double &B, set<uint> &I, double &initial_obj)
{
    groups = data.get_groups();             // store non selected nodes in each group
    select_count = vector<uint>(groups.size(), 0); // count # selected nodes in each group

    B = 0.0;
    I.clear(); // intact group indices
    for (int i = 0; i < budgets.size(); ++i)
    {
        if (budgets[i] > 0 && budgets[i] < groups[i].size())
        {
            I.emplace(i);
            B += budgets[i];
        }
        else // budgets[i] >= groups[i].size()
        {
            spdlog::info("Group {} has size {} less than budget {}",
                         i, groups[i].size(), budgets[i]);
            for (auto const &e : groups[i])
            {
                seeds[e] = true;
            }
            select_count[i] = groups[i].size();
        }
    }

    spdlog::info("B = {}", B);
    spdlog::info("Intitial I's size {}", I.size());

    initial_obj = query(seeds);
    spdlog::info("Initial objective: {}", initial_obj);
}

double Algorithm::query(const vector<bool> &s)
{
#pragma omp critical
    {
        ++num_queries;
    }

    if (Constants::APPLICATION == APP_TYPE::Video)
    {
        auto re = data.get_det_submatrix(s);
        if (Constants::LOG_OBJ)
            re = log(re);
        return re;
    }

    return data.get_no_active_nodes(s);
}

void Algorithm::reset_num_queries()
{
    num_queries = 0;
}