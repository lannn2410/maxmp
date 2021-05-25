#include "MpCommon.h"
#include <algorithm>
#include <numeric>
#include "Constants.h"

MpCommon *MpCommon::instance = nullptr;

MpCommon::MpCommon()
{
	seed = new int[10000];
	for (int i = 0; i < 10000; i++)
	{
		seed[i] = rand();
	}
}

MpCommon *MpCommon::getInstance()
{
	if (!instance)
		instance = new MpCommon();
	return instance;
}

unsigned MpCommon::randomInThread(int thread_id)
{
	unsigned tmp = seed[thread_id % 10000];
	tmp = tmp * 17931 + 7391;
	seed[thread_id % 10000] = tmp;
	return tmp;
}

set<uint> MpCommon::random_k_select(set<uint> S, const uint &k)
{
	if (k >= S.size())
	{
		return S;
	}

	set<uint> re{};
	for (int i = 0; i < k; ++i)
	{
		auto r = randomInThread(omp_get_thread_num()) % S.size();
		auto it = std::begin(S);
		std::advance(it, r);
		re.emplace(*it);
		S.erase(*it);
	}
	return re;
}

uint MpCommon::get_element_by_index_from_set(const uint &index, const set<uint> S) const
{
	auto it = std::begin(S);
	std::advance(it, index);
	return *it;
}

uint MpCommon::weighted_select(const vector<double> &w, const uint &ex)
{
	const uint max_idx = std::max_element(w.begin(), w.end()) - w.begin();
	vector<double> normalized_w;
	for (auto i = 0; i < w.size(); ++i)
	{
		if (i == max_idx)
			normalized_w.emplace_back(1.0);
		else
			normalized_w.emplace_back(pow(w[i] / w[max_idx], ex));
	}

	double sum_w = accumulate(normalized_w.begin(), normalized_w.end(), 0.0);
	auto r = ((double)(randomInThread(omp_get_thread_num()) % 10000)) / 10000.0;
	r *= sum_w;
	double acc = 0.0;
	for (int i = 0; i < normalized_w.size(); ++i)
	{
		if (acc <= r && acc + normalized_w[i] >= r)
		{
			return i;
		}
		acc += normalized_w[i];
	}
	return 0;
}

double MpCommon::gaussian_kernel(const vector<unsigned char> &v1,
								 const vector<unsigned char> &v2) const
{
	if (v1.size() != v2.size())
	{
		throw "Error: 2 vectors have different size";
	}
	double dist_sq = 0.0;
	for (int i = 0; i < v1.size(); ++i)
	{
		// normalize
		auto tmp = ((double)v1[i]) / 255.0 - ((double)v2[i]) / 255.0;
		dist_sq += tmp * tmp;
	}
	return exp(-Constants::KERNEL_GAMMA * dist_sq);
}

double MpCommon::vector_product_kernel(const vector<unsigned char> &v1,
									   const vector<unsigned char> &v2) const
{
	if (v1.size() != v2.size())
	{
		throw "Error: 2 vectors have different size";
	}
	double dist_sq = 0.0;
	for (int i = 0; i < v1.size(); ++i)
	{
		dist_sq += ((double)v1[i]) / 255.0 * ((double)v2[i]) / 255.0;
	}
	return Constants::KERNEL_GAMMA * dist_sq;
}

double MpCommon::determinant(const vector<vector<double>> &matrix) const
{
	if (matrix.empty())
		return 0;
	if (matrix.size() != matrix[0].size())
		throw "Error: matrix height and width are different";

	if (matrix.size() == 1)
		return matrix[0][0];

	if (matrix.size() == 2)
		return ((matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1]));

	double det = 0.0;
	auto n = matrix.size();
#pragma omp parallel for
	for (int x = 0; x < n; x++)
	{
		vector<vector<double>> submatrix(n - 1, vector<double>(n - 1));
		int subi = 0;
		for (int i = 1; i < n; i++)
		{
			int subj = 0;
			for (int j = 0; j < n; j++)
			{
				if (j == x)
					continue;
				submatrix[subi][subj] = matrix[i][j];
				subj++;
			}
			subi++;
		}
		auto det_submatrix = determinant(submatrix);
#pragma omp critical
		{
			det = det + (pow(-1, x) * matrix[0][x] * det_submatrix);
		}
	}
	return det;
}

double MpCommon::fast_determinant(vector<vector<double>> mat) const
{
	if (mat.empty())
		return 0;

	if (mat.size() != mat[0].size())
		throw "Error: matrix height and width are different";

	if (mat.size() == 1)
		return mat[0][0];

	if (mat.size() == 2)
		return ((mat[0][0] * mat[1][1]) - (mat[1][0] * mat[0][1]));

	double num1, num2, det = 1; // Initialize result
	uint index;
	uint n = mat.size();

	// loop for traversing the diagonal elements
	for (int i = 0; i < n; i++)
	{
		index = i; // initialize the index

		// finding the index which has non zero value
		while (mat[index][i] == 0 && index < n)
		{
			index++;
		}
		if (index == n) // if there is non zero element
		{
			// the determinat of matrix as zero
			continue;
		}
		if (index != i)
		{
			// loop for swaping the diagonal element row and
			// index row
			for (int j = 0; j < n; j++)
			{
				swap(mat[index][j], mat[i][j]);
			}
			// determinant sign changes when we shift rows
			// go through determinant properties
			det = det * pow(-1, index - i);
		}

		// storing the values of diagonal row elements
		auto const &temp = mat[i];

		// traversing every row below the diagonal element
		for (int j = i + 1; j < n; j++)
		{
			num1 = temp[i];	  // value of diagonal element
			num2 = mat[j][i]; // value of next row element

			// traversing every column of row
			// and multiplying to every row
			for (int k = 0; k < n; k++)
			{
				// multiplying to make the diagonal
				// element and next row element equal
				mat[j][k] -= (num2 * temp[k] / num1);
			}
		}
	}

	// mulitplying the diagonal elements to get determinant
	for (int i = 0; i < n; i++)
	{
		det = det * mat[i][i];
	}

	return det; // Det(kA)/k=Det(A);
}