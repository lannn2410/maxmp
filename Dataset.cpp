#include "Dataset.h"
#include <iostream>
#include <fstream>
#include <algorithm> // std::sort, std::stable_sort
#include <queue>
#include <stdio.h>
#include <sstream>

#include "Constants.h"

Dataset::Dataset()
{
}

void Dataset::read_network(const int &num_nodes, const string &file_name,
						   const int &num_seeds)
{
	clear();
	this->num_nodes = num_nodes;
	neighbors = vector<set<uint>>(num_nodes);

	ifstream is(file_name);
	is.seekg(0, is.end);
	long bufSize = is.tellg();
	is.seekg(0, is.beg);
	int item = 0;

	char *buffer = new char[bufSize];

	is.read(buffer, bufSize);
	is.close();

	std::string::size_type sz = 0;
	long sp = 0;
	uint start_id, end_id;
	bool is_start = true;
	uint id = 0;
	uint s_id, e_id; // used to stored ordered id of startId and endId
	uint edge_id = 0;

	while (sp < bufSize)
	{
		char c = buffer[sp];
		item = item * 10 + c - 48;
		sp++;
		if (sp == bufSize || (sp < bufSize && (buffer[sp] < 48 || buffer[sp] > 57)))
		{
			while (sp < bufSize && (buffer[sp] < 48 || buffer[sp] > 57))
				sp++;

			if (is_start)
			{
				start_id = item;
				is_start = false;
			}
			else
			{
				end_id = item;
				is_start = true;

				if (start_id != end_id)
				{
					auto const s_id_it = map_node_id.find(start_id);
					if (s_id_it == map_node_id.end())
					{
						map_node_id[start_id] = id;
						s_id = id;
						id++;
					}
					else
					{
						s_id = s_id_it->second;
					}

					auto const e_id_it = map_node_id.find(end_id);
					if (e_id_it == map_node_id.end())
					{
						map_node_id[end_id] = id;
						e_id = id;
						id++;
					}
					else
					{
						e_id = e_id_it->second;
					}

					// undirected graph
					neighbors[s_id].emplace(e_id);
					neighbors[e_id].emplace(s_id);
				}
			}
			item = 0;
		}
	}

	// get nodes of highest degree to form seed set
	vector<uint> idx(num_nodes);
	for (int i = 0; i < num_nodes; ++i)
	{
		idx[i] = i;
	}
	sort(idx.begin(), idx.end(),
		 [this](size_t i1, size_t i2) {
			 return this->neighbors[i1].size() > this->neighbors[i2].size();
		 });
	for (int i = 0; i < num_seeds; ++i)
	{
		seeds.emplace(idx[i]);
		spdlog::info("Seed {} degree {}", idx[i], neighbors[idx[i]].size());
	}

	spdlog::info("Finish reading graph of {} file, {} nodes", file_name, num_nodes);

	sample_graph_realization(Constants::NUM_SAMPLES);
	spdlog::info("Finish sampling {} graph realizations", Constants::NUM_SAMPLES);

	delete[] buffer;
}

double Dataset::get_no_active_nodes(const vector<bool> &boosted_nodes) const
{
	double re = 0.0;

#pragma omp parallel for
	for (int i = 0; i < sample_graphs.size(); ++i)
	{
		auto const &sample = sample_graphs[i];
		uint no_active_nodes = 0;
		// BFS
		queue<uint> q;
		vector<bool> checked(num_nodes, false);
		for (auto const &s : seeds)
		{
			q.push(s);
			checked[s] = true;
		}

		while (!q.empty())
		{
			auto const u = q.front();
			++no_active_nodes;
			q.pop();
			auto const &sample_u = sample[u];
			for (auto const &nei : sample_u.out_neighbors)
			{
				if (!checked[nei])
				{
					q.push(nei);
					checked[nei] = true;
				}
			}
			for (auto const &boost_nei : sample_u.boost_out_neighbors)
			{
				if (!checked[boost_nei] && boosted_nodes[boost_nei])
				{
					q.push(boost_nei);
					checked[boost_nei] = true;
				}
			}
		}

#pragma omp critical
		{
			re += no_active_nodes;
		}
	}

	return re / sample_graphs.size();
}

vector<set<uint>> Dataset::get_groups() const
{
	return groups;
}

void Dataset::clear()
{
	groups.clear();

	map_node_id.clear();
	neighbors.clear();
	seeds.clear();
	sample_graphs.clear();

	gram_matrix.clear();
}

void Dataset::sample_graph_realization(const int &num_samples)
{
	auto common_ins = MpCommon::getInstance();

#pragma omp parallel for
	for (int i = 0; i < num_samples; ++i)
	{
		vector<SampleNode> sample = vector<SampleNode>(num_nodes);
		for (int u = 0; u < num_nodes; ++u)
		{
			// sample each node
			for (auto const &out_neighbor : neighbors[u])
			{
				auto const r = common_ins->randomInThread(omp_get_thread_num()) % neighbors[out_neighbor].size();

				if (r < Constants::PROB_NO_BOOST)
				{
					sample[u].out_neighbors.emplace(out_neighbor);
				}
				else if (r < Constants::PROB_BOOST)
				{
					sample[u].boost_out_neighbors.emplace(out_neighbor);
				}
			}
		}
#pragma omp critical
		{
			sample_graphs.emplace_back(sample);
		}
	}

	spdlog::info("Finish generating {} samples", num_samples);
}

void Dataset::read_video(const string &video_name, const uint &width,
						 const uint &height)
{
	clear();
	// unsigned char frame[height][width][3] = {0};
	auto const rb = height * width * 3;
	unsigned char frame[rb] = {0};
	std::stringstream s;
	s << "ffmpeg -i " << video_name
	  << " -r 1/1 -f image2pipe -vcodec rawvideo -pix_fmt rgb24 -";
	FILE *pipein = popen(s.str().c_str(), "r");
	vector<vector<unsigned char>> frame_vecs;
	while (1)
	{
		// Read a frame from the input pipe into the buffer
		auto count = fread(frame, 1, rb, pipein);

		// If we didn't get a frame of video, we're probably at the end
		if (count != rb)
			break;
		vector<unsigned char> vec(frame, frame + sizeof(frame) / sizeof(frame[0]));
		frame_vecs.emplace_back(vec);
	}

	// Flush and close input and output pipes
	fflush(pipein);
	pclose(pipein);

	// compute gram matrix
	gram_matrix = vector<vector<double>>(frame_vecs.size(),
										 vector<double>(frame_vecs.size(), 1.0));
	for (int i = 0; i < frame_vecs.size(); ++i)
	{
#pragma omp parallel for
		for (int j = i; j < frame_vecs.size(); ++j)
		{
			double kernel_val = 0.0;
			switch (Constants::KERNEL)
			{
			case KERNEL_TYPE::VECTOR_PRODUCT:
				kernel_val = MpCommon::getInstance()->vector_product_kernel(
					frame_vecs[i], frame_vecs[j]);
				break;
			default:
				kernel_val = MpCommon::getInstance()->gaussian_kernel(
					frame_vecs[i], frame_vecs[j]);
				break;
			}
			if (i == j)
				kernel_val += 1;
			gram_matrix[i][j] = kernel_val;
			gram_matrix[j][i] = kernel_val;
			spdlog::info("Gram matrix: {} {} {}", i, j, kernel_val);
		}
		spdlog::info("Gram matrix: finish {} rows", i);
	}
	spdlog::info("Finish reading video");
}

double Dataset::get_det_submatrix(const vector<bool> &indices) const
{
	vector<uint> idx{};
	for (int i = 0; i < indices.size(); ++i)
	{
		if (indices[i])
			idx.emplace_back(i);
	}

	if (idx.empty())
		return 1.0;

	vector<vector<double>> submatrix(idx.size(), vector<double>(idx.size(), 0.0));
	// 2.0 since matrix will be I + X
	for (int i = 0; i < idx.size(); ++i)
	{
		for (int j = i; j < idx.size(); ++j)
		{
			submatrix[i][j] = gram_matrix[idx[i]][idx[j]];
			submatrix[j][i] = submatrix[i][j];
		}
	}
	// return MpCommon::getInstance()->determinant(submatrix);
	return MpCommon::getInstance()->fast_determinant(submatrix);
}

uint Dataset::get_data_size() const
{
	if (Constants::APPLICATION == APP_TYPE::Video)
		return gram_matrix.size();

	return num_nodes;
}

uint Dataset::get_num_groups() const
{
	return groups.size();
}

void Dataset::form_groups(const uint &num_groups)
{
	groups.clear();
	groups = vector<set<uint>>(num_groups);

	if (Constants::APPLICATION == APP_TYPE::Video)
	{
		// fill groups
		const double n = gram_matrix.size();
		uint const frame_per_group = ceil(n / num_groups);
		uint group_idx = 0;
		for (int i = 0; i < n; ++i)
		{
			groups[group_idx].insert(i);
			if (groups[group_idx].size() == frame_per_group)
				++group_idx;
		}
		spdlog::info("Finish forming {} groups of frames, {} frames per group",
					 num_groups, frame_per_group);
		return;
	}

	for (int i = 0; i < num_nodes; ++i)
	{
		auto g_idx = rand() % num_groups;
		groups[g_idx].insert(i);
	}

	spdlog::info("Finish forming {} groups of nodes", num_groups);
	for (int i = 0; i < num_groups; ++i)
	{
		spdlog::info("Group {} contains {} nodes", i, groups[i].size());
	}
}