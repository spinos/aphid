/*
 *  GridSampler.h
 *  
 *
 *  Created by jian zhang on 2/11/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_GRID_SAMPLER_H
#define APH_SDB_GRID_SAMPLER_H

#include <math/BoundingBox.h>

namespace aphid {

namespace sdb {

template<typename Tf, typename Tn, int Ndiv>
class GridSampler {

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
		stat = fintersect.selectedClosestToPoint(childBx.center(), childBx.radius() );
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
