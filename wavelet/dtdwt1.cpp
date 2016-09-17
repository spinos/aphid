/*
 *  dtdwt1.cpp
 *  
 *
 *  Created by jian zhang on 9/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "dtdwt1.h"

namespace aphid {

namespace wla {

DualTree::DualTree() :
m_numStages(0)
{}

DualTree::~DualTree()
{}
	
void DualTree::analize(const VectorN<float> & x, const int & nstage)
{
	m_numStages = 0;

	const int & n = x.N();
	VectorN<float> Xshf;
	Xshf.copy(x, -5);
/// normalize
	Xshf *= 0.707106781187f;
	
	VectorN<float> xu;
	xu.create(n);
	VectorN<float> xd;
	xd.create(n);
	
	int j = 0; 
	
	afbflt(Xshf.v(), n, xu, m_w[0][0], Dtf::FirstStageUpFarrasAnalysis);
	afbflt(Xshf.v(), n, xd, m_w[0][1], Dtf::FirstStageDownFarrasAnalysis);
	
	j++;
	
	while(j<nstage && j<(DT_MAX_N_STAGE-1) ) {
	
		const int nj = xu.N();
		
		Xshf.copy(xu, -5);

		afbflt(Xshf.v(), nj, xu, m_w[j][0], Dtf::UpAnalysis);
			
		Xshf.copy(xd, -5);

		afbflt(Xshf.v(), nj, xd, m_w[j][1], Dtf::DownAnalysis);
		
		j++;
		
	}
	
	m_w[j][0] = xu;
	m_w[j][1] = xd;
	
	m_numStages = j;
}

void DualTree::synthesize(VectorN<float> & y)
{
	int j = m_numStages;
	VectorN<float> yu;
	yu.copy(m_w[j][0]);
	VectorN<float> yd;
	yd.copy(m_w[j][1]);

	j--;
	while(j>0) {
		
		sfbflt(yu, yu, m_w[j][0], Dtf::UpSynthesis);
		sfbflt(yd, yd, m_w[j][1], Dtf::DownSynthesis);
		
		j--;
	}
	
	sfbflt(yu, yu, m_w[j][0], Dtf::FirstStageUpFarrasSynthesis);
	sfbflt(yd, yd, m_w[j][1], Dtf::FirstStageDownFarrasSynthesis);
	
	const int & n = yu.N();
	y.copy(yu);
	for(int i=0; i<n;++i) {
		y[i] += yd[i];
		y[i] *= 0.707106781187f;
	}
	
}
	
const int & DualTree::numStages() const
{ return m_numStages; }

const VectorN<float> & DualTree::stage(const int & i, const int & j ) const
{ return m_w[i][j]; }

}

}