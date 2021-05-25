#include "FastThreshold.h"

#include <numeric> // std::accumulate
#include <math.h>

FastThreshold::FastThreshold(const Dataset &d, const vector<uint> &budgets)
    : Algorithm(d, budgets)
{
}

// double FastThreshold::get_solutions(vector<bool> &seeds)
// {
//     spdlog::info("Start Fast Threshold Algorithm");
//     vector<set<uint>> groups{};
//     vector<uint> select_count;
//     double B = 0.0, current_obj = 0.0;
//     set<uint> I{};
//     initiate(seeds, groups, select_count, B, I, current_obj);

//     // lambda function to get max_gain
//     auto get_max_gain = [this, &current_obj, &I, &groups, &seeds]() -> double {
//         double re = 0.0;
//         // auto copy_seeds = seeds;
//         for (auto const &g_idx : I)
//         {
// #pragma omp parallel for
//             for (int i = 0; i < groups[g_idx].size(); ++i)
//             {
//                 // auto it = std::begin(groups[g_idx]);
//                 // std::advance(it, i);
//                 auto const e = this->common_ins->get_element_by_index_from_set(i, groups[g_idx]);
//                 auto copy_seed = seeds;
//                 copy_seed[e] = true;
//                 auto tmp = query(copy_seed) - current_obj;
//                 if (tmp > re)
//                 {
// #pragma omp critical
//                     {
//                         if (tmp > re)
//                         {
//                             re = tmp;
//                         }
//                     }
//                 }
//             }
//         }
//         return re;
//     };

//     // initiate tau and tao0
//     double tau = get_max_gain();
//     spdlog::info("Initiate tau = {}", tau);

//     auto lower_tau = Constants::FAST_THRESHOLD_EPSILON *
//                      (1.0 - Constants::FAST_THRESHOLD_EPSILON) * tau / B;
//     while (!I.empty() && tau >= lower_tau)
//     {
//         spdlog::info("{} groups are not full", I.size());
//         set<uint> full_group_indices{}; // store group indices that will be full after this loop

//         if (Constants::FAST_THRESHOLD_SOFT_GREEDY)
//         {
//             auto tau1 = get_max_gain();
//             if (tau1 < tau)
//             {
//                 tau = tau1;
//                 spdlog::info("tau reduced to {} with new max gain", tau);
//             }
//         }

//         for (auto const &g_idx : I)
//         {
//             set<uint> new_elements{}; // store new elements put to group
//             uint start_idx = 0;
//             while (start_idx < groups[g_idx].size())
//             {
//                 uint min_idx = groups[g_idx].size();
//                 double save_obj = 0.0;
//                 uint flag_stop = groups[g_idx].size();
// #pragma omp parallel for
//                 for (int i = start_idx; i < flag_stop; ++i)
//                 {
//                     // auto it = std::begin(groups[g_idx]);
//                     // std::advance(it, i);
//                     auto const e = common_ins->get_element_by_index_from_set(i, groups[g_idx]);
//                     auto copy_seeds = seeds;
//                     if (copy_seeds[e])
//                     {
//                         spdlog::info("Error: not erase selected elements {}", e);
//                         throw "Error";
//                     }
//                     copy_seeds[e] = true;
//                     auto tmp_obj = query(copy_seeds);
//                     if (tmp_obj - current_obj >= tau && i < min_idx)
//                     {
// #pragma omp critical
//                         {
//                             if (i < min_idx)
//                             {
//                                 save_obj = tmp_obj;
//                                 min_idx = i;
//                                 flag_stop = 0; // a way to break for loop in openmp
//                             }
//                         }
//                     }
//                 }

//                 if (min_idx == groups[g_idx].size())
//                     break;

//                 auto const min_e = common_ins->get_element_by_index_from_set(min_idx, groups[g_idx]);
//                 seeds[min_e] = true;
//                 current_obj = save_obj;
//                 spdlog::info("Element {} is selected to put to group {}, new objective {}",
//                              min_e, g_idx, current_obj);
//                 new_elements.emplace(min_e);
//                 ++select_count[g_idx];
//                 if (select_count[g_idx] >= budgets[g_idx])
//                 {
//                     full_group_indices.emplace(g_idx);
//                     spdlog::info("Group {} is full", g_idx);
//                     break;
//                 }

//                 start_idx = min_idx + 1;
//             }

//             // erase elements that were selected
//             for (auto const &e : new_elements)
//             {
//                 groups[g_idx].erase(e);
//             }
//         }
//         // erase groups that were full
//         for (auto const &full_g_idx : full_group_indices)
//         {
//             I.erase(full_g_idx);
//         }

//         tau *= (1.0 - Constants::FAST_THRESHOLD_EPSILON);
//         spdlog::info("tau reduced to {} after loop", tau);
//     }

//     spdlog::info("Final objective: {}", current_obj);
//     spdlog::info("End Fast Threshold Algorithm");
//     return current_obj;
// }

double FastThreshold::get_solutions(vector<bool> &seeds)
{
    spdlog::info("Start Fast Threshold Algorithm");
    vector<set<uint>> groups{};
    vector<uint> select_count;
    double B = 0.0, current_obj = 0.0;
    set<uint> I{};
    initiate(seeds, groups, select_count, B, I, current_obj);

    // lambda function to get max_gain
    auto get_max_gain = [this, &current_obj, &I, &groups, &seeds]() -> double {
        double re = 0.0;
        // auto copy_seeds = seeds;
        for (auto const &g_idx : I)
        {
#pragma omp parallel for
            for (int i = 0; i < groups[g_idx].size(); ++i)
            {
                // auto it = std::begin(S);
                // std::advance(it, i);
                // auto const &e = *it;
                auto const e = this->common_ins->get_element_by_index_from_set(i, groups[g_idx]);
                auto copy_seed = seeds;
                copy_seed[e] = true;
                auto tmp = query(copy_seed) - current_obj;
                if (tmp > re)
                {
#pragma omp critical
                    {
                        if (tmp > re)
                        {
                            re = tmp;
                        }
                    }
                }
            }
        }
        return re;
    };

    // initiate tau and tao0
    double tau = get_max_gain();
    spdlog::info("Initiate tau = {}", tau);

    auto lower_tau = Constants::FAST_THRESHOLD_EPSILON *
                     (1.0 - Constants::FAST_THRESHOLD_EPSILON) * tau / B;
    while (!I.empty() && tau >= lower_tau)
    {
        spdlog::info("{} groups are not full", I.size());
        set<uint> full_group_indices{}; // store group indices that will be full after this loop

        if (Constants::FAST_THRESHOLD_SOFT_GREEDY)
        {
            auto tau1 = get_max_gain();
            if (tau1 < tau)
            {
                tau = tau1;
                spdlog::info("tau reduced to {} with new max gain", tau);
            }
        }

        for (auto const &g_idx : I)
        {
            set<uint> new_elements{}; // store new elements put to group
            for (auto const &e : groups[g_idx])
            {
                auto copy_seeds = seeds;
                if (copy_seeds[e])
                {
                    spdlog::info("Error: not erase selected elements");
                    throw "Error";
                }
                copy_seeds[e] = true;
                auto tmp_obj = query(copy_seeds);
                if (tmp_obj - current_obj >= tau)
                {
                    seeds[e] = true;
                    current_obj = tmp_obj;
                    spdlog::info("Element {} is selected to put to group {}, new objective {}",
                                 e, g_idx, current_obj);
                    new_elements.emplace(e);
                    ++select_count[g_idx];
                    if (select_count[g_idx] >= budgets[g_idx])
                    {
                        full_group_indices.emplace(g_idx);
                        spdlog::info("Group {} is full", g_idx);
                        break;
                    }
                }
            }
            // erase elements that were selected
            for (auto const &e : new_elements)
            {
                groups[g_idx].erase(e);
            }
        }
        // erase groups that were full
        for (auto const &full_g_idx : full_group_indices)
        {
            I.erase(full_g_idx);
        }

        tau *= (1.0 - Constants::FAST_THRESHOLD_EPSILON);
        spdlog::info("tau reduced to {} after loop", tau);
    }

    spdlog::info("Final objective: {}", current_obj);
    spdlog::info("End Fast Threshold Algorithm");
    return current_obj;
}