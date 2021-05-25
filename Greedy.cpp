#include "Greedy.h"

#include <numeric> // std::accumulate
#include <math.h>

Greedy::Greedy(const Dataset &d, const vector<uint> &budgets)
    : Algorithm(d, budgets)
{
}

double Greedy::get_solutions(vector<bool> &seeds)
{
    spdlog::info("Start Greedy Algorithm");

    // double current_obj = query(seeds);
    // spdlog::info("Initial objective: {}", current_obj);

    // auto groups = data.get_groups();             // store non selected nodes in each group
    // vector<uint> select_count(groups.size(), 0); // count # selected nodes in each group

    // set<uint> I; // intact group indices
    // for (int i = 0; i < budgets.size(); ++i)
    // {
    //     if (budgets[i] > 0)
    //         I.emplace(i);
    // }
    vector<set<uint>> groups{};
    vector<uint> select_count;
    double B = 0.0, current_obj = 0.0;
    set<uint> I{};
    initiate(seeds, groups, select_count, B, I, current_obj);

    while (!I.empty())
    {
        spdlog::info("{} groups are not full", I.size());

        uint sel_group_idx, sel_e;
        double max_obj = 0.0;
        // auto copy_seeds = seeds;
        for (auto const &g_idx : I)
        {
            auto const &gr = groups[g_idx];
#pragma omp parallel for
            for (int i = 0; i < gr.size(); ++i)
            {
                auto it = std::begin(gr);
                std::advance(it, i);
                auto const &e = *it;
                auto copy_seeds = seeds;
                if (copy_seeds[e])
                {
                    spdlog::info("Error: not erase selected elements");
                    throw "Error";
                }
                copy_seeds[e] = true;
                auto tmp_obj = query(copy_seeds);
                if (tmp_obj > max_obj)
                {
#pragma omp critical
                    {
                        if (tmp_obj > max_obj)
                        {

                            max_obj = tmp_obj;
                            sel_group_idx = g_idx;
                            sel_e = e;
                        }
                    }
                }
            }

            // for (auto const &e : groups[g_idx])
            // {
            //     if (copy_seeds[e])
            //     {
            //         spdlog::info("Error: not erase selected elements");
            //         throw "Error";
            //     }
            //     copy_seeds[e] = true;
            //     auto tmp_obj = query(copy_seeds);
            //     if (tmp_obj > max_obj)
            //     {
            //         max_obj = tmp_obj;
            //         sel_group_idx = g_idx;
            //         sel_e = e;
            //     }
            //     copy_seeds[e] = false;
            // }
        }
        if (max_obj < current_obj)
            break;

        seeds[sel_e] = true;
        current_obj = max_obj;
        groups[sel_group_idx].erase(sel_e);
        spdlog::info("Element {} is selected to put to group {}, new objective {}",
                     sel_e, sel_group_idx, max_obj);

        ++select_count[sel_group_idx];
        if (select_count[sel_group_idx] >= budgets[sel_group_idx])
        {
            spdlog::info("Group {} is full", sel_group_idx);
            I.erase(sel_group_idx);
        }
    }

    spdlog::info("Final objective: {}", current_obj);
    spdlog::info("End Greedy Algorithm");
    return current_obj;
}