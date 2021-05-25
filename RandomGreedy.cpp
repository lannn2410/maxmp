#include "RandomGreedy.h"

#include <numeric> // std::accumulate
#include <math.h>

RandomGreedy::RandomGreedy(const Dataset &d, const vector<uint> &budgets)
    : Algorithm(d, budgets)
{
}

double RandomGreedy::get_solutions(vector<bool> &seeds)
{
    spdlog::info("Start Random Greedy Algorithm");

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

        vector<uint> max_elements, max_element_to_group;
        vector<double> max_element_objs;

        for (auto const &g_idx : I)
        {
            auto const &gr = groups[g_idx];
            vector<uint> elements(gr.size(), 0);
            vector<double> element_objs(gr.size(), 0.0);

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
                elements[i] = e;
                element_objs[i] = tmp_obj;
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
            //     elements.emplace_back(e);
            //     element_objs.emplace_back(tmp_obj);
            //     copy_seeds[e] = false;
            // }

            // take out B_i - |S_i| elements with max objs
            vector<uint> e_idx(groups[g_idx].size(), 0);
            for (int i = 0; i < groups[g_idx].size(); ++i)
            {
                e_idx[i] = i;
            }
            sort(e_idx.begin(), e_idx.end(), [&element_objs](size_t i1, size_t i2) {
                return element_objs[i1] > element_objs[i2];
            });
            for (int i = 0; i < budgets[g_idx] - select_count[g_idx]; ++i)
            {
                // spdlog::info("e_idx[{}] = {}", i, e_idx[i]);
                // spdlog::info("elements[{}] = {}", e_idx[i], elements[e_idx[i]]);
                max_elements.emplace_back(elements[e_idx[i]]);
                max_element_to_group.emplace_back(g_idx);
                max_element_objs.emplace_back(element_objs[e_idx[i]]);
            }
        }

        if (max_elements.empty())
            break;

        auto sel_idx = common_ins->randomInThread(omp_get_thread_num()) %
                       max_elements.size();

        auto const &sel_e = max_elements[sel_idx];
        auto const &max_obj = max_element_objs[sel_idx];
        auto const &sel_group_idx = max_element_to_group[sel_idx];

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
    spdlog::info("End Random Greedy Algorithm");
    return current_obj;
}