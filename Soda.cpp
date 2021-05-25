#include "Soda.h"

#include <numeric> // std::accumulate
#include <math.h>

Soda::Soda(const Dataset &d, const vector<uint> &budgets)
    : Algorithm(d, budgets)
{
    p = Constants::SODA_BETA /
        (Constants::SODA_BETA +
         sqrt(Constants::SODA_BETA * (1 - Constants::SODA_BETA)));
}

double Soda::get_solutions(vector<bool> &seeds)
{
    spdlog::info("Start Soda Algorithm");

    vector<set<uint>> groups{};
    vector<uint> select_count;
    double B = 0.0, current_obj = 0.0;
    set<uint> I{};
    initiate(seeds, groups, select_count, B, I, current_obj);

    vector<uint> select_count_A(select_count), select_count_B(select_count),
        select_count_A_B(select_count);
    vector<bool> seed_A(seeds), seed_B(seeds), seed_A_B(seeds);
    auto obj_A = current_obj, obj_B = current_obj;
    auto I_A = I, I_B = I;
    auto groups_A = groups, groups_B = groups;

    split(seed_A, seed_B, seed_A_B,
          select_count_A, select_count_B, select_count_A_B,
          obj_A, obj_B, I_A, I_B, groups_A, groups_B, I, groups);

    spdlog::info("Start res. greedy for A, obj {}", obj_A);
    residual_greedy(seed_A, obj_A, select_count_A, I_A, groups_A);
    spdlog::info("Done res. greedy for A, obj {}", obj_A);

    spdlog::info("Start res. greedy for B, obj {}", obj_B);
    residual_greedy(seed_B, obj_B, select_count_B, I_B, groups_B);
    spdlog::info("Done res. greedy for B, obj {}", obj_B);

    if (obj_A > obj_B)
    {
        current_obj = obj_A;
        seeds = seed_A;
    }
    else
    {
        current_obj = obj_B;
        seeds = seed_B;
    }

    spdlog::info("Final objective: {}", current_obj);
    spdlog::info("End Soda Algorithm");
    return current_obj;
}

void Soda::split(vector<bool> &seed_A, vector<bool> &seed_B,
                 vector<bool> &seed_A_B,
                 vector<uint> &select_count_A, vector<uint> &select_count_B,
                 vector<uint> &select_count_A_B,
                 double &obj_A, double &obj_B,
                 set<uint> &I_A, set<uint> &I_B,
                 vector<set<uint>> &groups_A, vector<set<uint>> &groups_B,
                 set<uint> I, vector<set<uint>> groups)
{
    while (!I.empty())
    {
        spdlog::info("{} groups are not full", I.size());
        uint sel_group_idx_A, sel_group_idx_B, sel_e_A, sel_e_B;
        double max_obj_A = 0.0, max_obj_B = 0.0;
        uint sel_group_idx, sel_e;
        for (auto const &g_idx : I)
        {
            auto const &gr = groups[g_idx];
#pragma omp parallel for
            for (int i = 0; i < gr.size(); ++i)
            {
                auto const e = common_ins->get_element_by_index_from_set(i, gr);
                auto copy_seed_A = seed_A, copy_seed_B = seed_B;
                if (copy_seed_A[e] || copy_seed_B[e] || seed_A_B[e])
                {
                    spdlog::info("Error: not erase selected elements");
                    throw "Error";
                }
                copy_seed_A[e] = true;
                copy_seed_B[e] = true;
                auto tmp_obj_A = query(copy_seed_A);
                auto tmp_obj_B = query(copy_seed_B);
                if (tmp_obj_A > max_obj_A)
                {
#pragma omp critical
                    {
                        if (tmp_obj_A > max_obj_A)
                        {

                            max_obj_A = tmp_obj_A;
                            sel_group_idx_A = g_idx;
                            sel_e_A = e;
                        }
                    }
                }

                if (tmp_obj_B > max_obj_B)
                {
#pragma omp critical
                    {
                        if (tmp_obj_B > max_obj_B)
                        {

                            max_obj_B = tmp_obj_B;
                            sel_group_idx_B = g_idx;
                            sel_e_B = e;
                        }
                    }
                }
            }
        }

        auto gain_A = max_obj_A - obj_A;
        auto gain_B = max_obj_B - obj_B;
        if (p * gain_A >= (1.0 - p) * gain_B)
        {
            sel_e = sel_e_A;
            sel_group_idx = sel_group_idx_A;
            ++select_count_A[sel_group_idx];
            obj_A = max_obj_A;
            seed_A[sel_e_A] = true;
            groups_A[sel_group_idx].erase(sel_e);
            if (select_count_A[sel_group_idx] >= budgets[sel_group_idx])
            {
                spdlog::info("Split Step: A - Group {} is full", sel_group_idx);
                I_A.erase(sel_group_idx);
            }
            spdlog::info("Split Step: A - Element {} is selected to put to group {}, new obj {}",
                         sel_e, sel_group_idx, obj_A);
        }
        else
        {
            sel_e = sel_e_B;
            sel_group_idx = sel_group_idx_B;
            ++select_count_B[sel_group_idx];
            obj_B = max_obj_B;
            seed_B[sel_e_B] = true;
            groups_B[sel_group_idx].erase(sel_e);
            if (select_count_B[sel_group_idx] >= budgets[sel_group_idx])
            {
                spdlog::info("Split Step: B - Group {} is full", sel_group_idx);
                I_B.erase(sel_group_idx);
            }
            spdlog::info("Split Step: B - Element {} is selected to put to group {}, new obj {}",
                         sel_e, sel_group_idx, obj_B);
        }

        seed_A_B[sel_e] = true;
        groups[sel_group_idx].erase(sel_e);
        spdlog::info("Split Step: Element {} is selected to put to group {}",
                     sel_e, sel_group_idx);
        ++select_count_A_B[sel_group_idx];
        if (select_count_A_B[sel_group_idx] >= budgets[sel_group_idx])
        {
            spdlog::info("Split Step: Group {} is full", sel_group_idx);
            I.erase(sel_group_idx);
        }
    }
    spdlog::info("Done Split step");
}

void Soda::residual_greedy(vector<bool> &seeds, double &obj,
                           vector<uint> &select_count,
                           set<uint> &I, vector<set<uint>> &groups)
{
    while (!I.empty())
    {
        spdlog::info("Res. Greedy: {} groups are not full", I.size());

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
                auto const e = common_ins->get_element_by_index_from_set(i, gr);
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
        obj = max_obj;
        groups[sel_group_idx].erase(sel_e);
        spdlog::info("Res. Greedy: Element {} is selected to put to group {}, new objective {}",
                     sel_e, sel_group_idx, max_obj);

        ++select_count[sel_group_idx];
        if (select_count[sel_group_idx] >= budgets[sel_group_idx])
        {
            spdlog::info("Res. Greedy: Group {} is full", sel_group_idx);
            I.erase(sel_group_idx);
        }
    }

    spdlog::info("Final objective: {}", obj);
    spdlog::info("End Res. Greedy Algorithm");
}