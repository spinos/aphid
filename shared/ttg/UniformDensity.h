/*
 *  UniformDensity.h
 *  
 *  estimate normal direction by density gradient
 *  uniform grid of M-by-N-by-P dimension
 *  densities are stored at cell centers
 *  at straggled cells find sign changes across 8 cells
 *  [0,7] solid cells
 *  0 1  0 1  0 1       0 1
 *  0 0  0 1  1 1 good  1 0 bad
 *  node at cell center, edges are axix-aligned and have same length
 *  node position and distance has no use
 *
 *  Created by jian zhang on 2/10/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_UNIFORM_DENSITY_H
#define APH_TTG_UNIFORM_DENSITY_H

#include <sdb/VectorArray.h>
#include <boost/scoped_array.hpp>
#include <graph/BaseDistanceField.h>

namespace aphid {

struct Float4;
class Vector3F;
class BoundingBox;

namespace ttg {

class UniformDensity : public BaseDistanceField {

/// (mean_position, density)
	boost::scoped_array<Float4> m_rho;

	struct DensityFront {
		float _nml[3];
		int _ind[3];
	};
/// where the boundary is
	boost::scoped_array<DensityFront> m_front;
	float m_subDensity[32];
	int m_numFronts;
/// (m,n,p,m*n*p)
	int m_dim[4];
/// (x,y,z,h,1/h)
	float m_originSize[5];
	
public:
	
	UniformDensity();
	virtual ~UniformDensity();
	
	void create(int M, int N, int P,
				const float* boxOrigin,
				const float& cellSize);
				
	void setOriginAndCellSize(const float* boxOrigin,
				const float& cellSize);
	
/// rho <- 0				
	void setZero();
/// accumulate in cells
	bool accumulate(const float& val, const Vector3F& p);
	void finish();

	bool isEmpty() const;
	const int& numCells() const;
	const int* dimension() const;
	const float& cellSize() const;
	const float& getDensity(int i, int j, int k) const;
	float safeGetDensity(int i, int j, int k) const;
	void safeGetPositionDensity(float* dst, int i, int j, int k) const;
	Vector3F getCellCenter(int i, int j, int k) const;
	BoundingBox getCellBox(int i, int j, int k) const;
	const int& numFronts() const;
	void getFront(BoundingBox& frontBx, aphid::Vector3F& frontNml, const int& i) const;
	
	template<typename T>
	void buildSamples(T& asamp, sdb::VectorArray<T>* dst);
	
private:
/// num_cell nodes
/// (M-1)NP + M(N-1)P + MN(P-1) edges
	void buildGraph();
/// at cell(i,j,k) origin
	bool detectFront(DensityFront* dst, int i, int j, int k);
/// by sud cell density delta
	void estimateNormal(DensityFront* dst) const;
	
	int cellInd(int i, int j, int k) const;
	
	int firstEmptyCellInd() const;
	
	void aggregateFrontSamplePos(Vector3F& pos, const DensityFront& fi);
	
	static const int EightSubCellCoord[8][3];
	
};

template<typename T>
void UniformDensity::buildSamples(T& asamp, sdb::VectorArray<T>* dst)
{
	asamp._r = cellSize() * .031f;
	for(int i=0;i<m_numFronts;++i) {
		const DensityFront& fi = m_front[i];
		asamp._nml = fi._nml;
		aggregateFrontSamplePos(asamp._pos, fi);
		asamp._pos += asamp._nml * asamp._r;
		dst->insert(asamp);
	}
}

}

}

#endif
