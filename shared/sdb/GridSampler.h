/*
 *  GridSampler.h
 *  
 *	Tf as closetToPointTest type, Tn as node type, Ndiv as order of grid
 *  Ndiv cells along each dimension
 *  test closet point in each cell 
 *  kmean to reduce number of samples 
 *
 *  Created by jian zhang on 2/11/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_GRID_SAMPLER_H
#define APH_SDB_GRID_SAMPLER_H

#include <math/BoundingBox.h>
#include <math/kmean.h>

namespace aphid {

namespace sdb {

template<typename Tf, typename Tn, int Ndiv>
class GridSampler {

	KMeansClustering2<float> m_cluster;
	DenseMatrix<float> m_data;
	Vector3F m_gridCoord[Ndiv * Ndiv * Ndiv];
	Tn m_samples[Ndiv * Ndiv * Ndiv];
	float m_cellDelta;
	int m_numValidSamples;

public:
	GridSampler();
	
	int numSamples() const;
	void sampleInBox(Tf & fintersect,
					const BoundingBox & box);
	const int & numValidSamples() const;
	const Tn & sample(const int & i) const;
	
private:
	void processKmean();
	
};

template<typename Tf, typename Tn, int Ndiv>
GridSampler<Tf, Tn, Ndiv>::GridSampler()
{
	m_cellDelta = 1.f / (float)Ndiv;
	int c = 0;
	for(int k=0;k<Ndiv;++k) {
		for(int j=0;j<Ndiv;++j) {
			for(int i=0;i<Ndiv;++i) {
				m_gridCoord[c].set(m_cellDelta * i, m_cellDelta * j, m_cellDelta * k); 
				c++;
			}
		}
	}
}

template<typename Tf, typename Tn, int Ndiv>
int GridSampler<Tf, Tn, Ndiv>::numSamples() const
{ return Ndiv * Ndiv * Ndiv; }

template<typename Tf, typename Tn, int Ndiv>
void GridSampler<Tf, Tn, Ndiv>::sampleInBox(Tf & fintersect,
					const BoundingBox & box)
{
	m_numValidSamples = 0;
	bool stat = fintersect.select(box);
	if(!stat) {
		return;
	}
	
	BoundingBox childBx;
	const int ns = numSamples();
	for(int i=0;i<ns;++i) {
		box.getSubBox(childBx, m_gridCoord[i], m_cellDelta);
		stat = fintersect.selectedClosestToPoint(childBx.center() );
		if(stat) {
			stat = childBx.isPointInside(fintersect.closestToPointPoint() );
		}
		if(stat) {
			Tn * d = &m_samples[m_numValidSamples];
			m_numValidSamples++;
			
			d->pos = fintersect.closestToPointPoint();
			d->nml = fintersect.closestToPointNormal();
			
		}
	}
	
	if(m_numValidSamples > 5) {
		processKmean();
	}
	
}

template<typename Tf, typename Tn, int Ndiv>
void GridSampler<Tf, Tn, Ndiv>::processKmean()
{
	const int n = m_numValidSamples;
	int k = n - 2;
	if(n > 12) {
		k = n - 1 - n / 3;
	}
	if(n > 24) {
		k = n / 2;
	}
	
/// position and normal
	const int d = 6;
/// to kmean data
	m_data.resize(n, d);
	for(int i=0;i<n;++i) {
		const Tn & src = m_samples[i];
		m_data.column(0)[i] = src.pos.x;
		m_data.column(1)[i] = src.pos.y;
		m_data.column(2)[i] = src.pos.z;
		m_data.column(3)[i] = src.nml.x * 2.5f;
		m_data.column(4)[i] = src.nml.y * 2.5f;
		m_data.column(5)[i] = src.nml.z * 2.5f;
	}
	
	m_cluster.setKND(k, n, d);
	if(!m_cluster.compute(m_data) ) {
		std::cout<<"\n GridSampler kmean failed ";
		return;
	}
/// from kmean data	
	DenseVector<float> centr;
	for(int i=0;i<k;++i) {
		m_cluster.getGroupCentroid(centr, i);
		Tn & dst = m_samples[i];
		dst.pos.set(centr[0], centr[1], centr[2]);
		dst.nml.set(centr[3], centr[4], centr[5]);
		dst.nml.normalize();
		
	}
	m_numValidSamples = k;
}

template<typename Tf, typename Tn, int Ndiv>
const int & GridSampler<Tf, Tn, Ndiv>::numValidSamples() const
{ return m_numValidSamples; }

template<typename Tf, typename Tn, int Ndiv>
const Tn & GridSampler<Tf, Tn, Ndiv>::sample(const int & i) const
{ return m_samples[i]; }

}

}
#endif
