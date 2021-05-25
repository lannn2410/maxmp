#pragma once
#include <string>
#include <vector>
#include <set>
#include <map>
#include "MpCommon.h"

using namespace std;

// use for sampling graph realization in social network
struct SampleNode
{
	set<uint> out_neighbors;	   // out neighbors if no boost
	set<uint> boost_out_neighbors; // additional out neighbors if boosted
};

class Dataset
{
public:
	Dataset();

	uint get_data_size() const;
	uint get_num_groups() const;
	void form_groups(const uint &num_groups);

	// for social network application
	void read_network(const int &num_nodes, const string &file_name,
					  const int &num_seeds);
	/* boosted nodes is represented under 0-1 vector for better performance*/
	double get_no_active_nodes(const vector<bool> &boosted_nodes) const;
	vector<set<uint>> get_groups() const;

	// for video summarization
	void read_video(const string &video_name, const uint &width,
					const uint &height);
	double get_det_submatrix(const vector<bool> &indices) const;

private:
	void clear();
	vector<set<uint>> groups; // group_id -> list of nodes

	// for social network application
	void sample_graph_realization(const int &num_samples);
	uint num_nodes;
	map<uint, uint> map_node_id;			  // map from true id -> ordered id (used for read graph from file)
	vector<set<uint>> neighbors;			  // node_id -> list of neighbors
	set<uint> seeds;						  // seed set, containing nodes of highest degree
	vector<vector<SampleNode>> sample_graphs; // pre-sample graph realization, for fast computation

	// for video summarization
	vector<vector<double>> gram_matrix;
};
