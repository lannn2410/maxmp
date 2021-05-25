#include "RandomizeAlg.h"

#include <numeric> // std::accumulate
#include <math.h>

RandomizeAlg::RandomizeAlg(const Dataset &d, const vector<uint> &budgets)
    : Algorithm(d, budgets)
{
}

double RandomizeAlg::get_solutions(vector<bool> &seeds)
{
    spdlog::info("Start Randomized Algorithm");

    vector<set<uint>> groups{};
    vector<uint> select_count;
    double B = 0.0, current_obj = 0.0;
    set<uint> I{};
    initiate(seeds, groups, select_count, B, I, current_obj);

    while (!I.empty())
    {
        spdlog::info("{} groups are not full", I.size());

        set<uint> full_group_indices{}; // store group indices that will be full after this loop
        for (auto const &g_idx : I)
        {
            spdlog::info("Consider group {}, remain size {}, selected {}",
                         g_idx, groups[g_idx].size(), select_count[g_idx]);

            uint k = ceil(groups[g_idx].size() / (budgets[g_idx] - select_count[g_idx]) *
                          log(B / Constants::RANDOMIZE_DELTA));
            spdlog::info("Will random pick {} elements from group {}", k, g_idx);

            auto k_select_nodes = common_ins->random_k_select(groups[g_idx], k);
            spdlog::info("Pick {} elements from group {}", k_select_nodes.size(), g_idx);

            if (k_select_nodes.empty())
            {
                spdlog::info("Error in random selection in group {}", g_idx);
                return 0;
            }

            vector<uint> map_node_idx(k_select_nodes.size(), 0);
            vector<double> weights(k_select_nodes.size(), 0.0), new_objs(k_select_nodes.size(), 0.0);

#pragma omp parallel for
            for (int i = 0; i < k_select_nodes.size(); ++i)
            {
                // auto it = std::begin(k_select_nodes);
                // std::advance(it, i);
                // auto const &select_node = *it;
                auto const select_node = common_ins->get_element_by_index_from_set(i, k_select_nodes);
                map_node_idx[i] = select_node;
                auto copy_seeds = seeds;
                if (copy_seeds[select_node])
                {
                    spdlog::info("Error: not erase selected elements");
                    throw "Error";
                }
                copy_seeds[select_node] = true;
                auto new_obj = query(copy_seeds);
                new_objs[i] = new_obj;
                double w = new_obj - current_obj;
                // spdlog::info("Element {} w {}", select_node, w);
                // weights[i] = pow(w, k_select_nodes.size() - 1);
                weights[i] = w;
                // spdlog::info("Weights of node {}: {}", select_node, weights[i]);
            }

            auto weighted_select_idx = common_ins->weighted_select(weights, k_select_nodes.size() - 1);
            // uint weighted_select_idx = std::max_element(weights.begin(), weights.end()) - weights.begin();
            auto new_e = map_node_idx[weighted_select_idx];
            seeds[new_e] = true;
            current_obj = new_objs[weighted_select_idx];
            spdlog::info("Element {} is selected to put to group {}, new objective {}",
                         new_e, g_idx, current_obj);

            groups[g_idx].erase(new_e);
            ++select_count[g_idx];
            if (select_count[g_idx] >= budgets[g_idx])
            {
                full_group_indices.emplace(g_idx);
                spdlog::info("Group {} is full", g_idx);
            }
        }

        // erase groups that are full
        for (auto const &full_g_idx : full_group_indices)
        {
            I.erase(full_g_idx);
        }
    }

    spdlog::info("Final objective: {}", current_obj);
    spdlog::info("End Randomized Algorithm");
    return current_obj;
}