/*
 *  dtdwt1.cpp
 *  
 *
 *  Created by jian zhang on 9/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "dtdwt2.h"

namespace aphid {

namespace wla {

DualTree2::DualTree2() :
m_numStages(0)
{}

DualTree2::~DualTree2()
{}
	
void DualTree2::analize(const VectorN<float> & x, const int & nstage)
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
	
	afbDtFsU(Xshf.v(), n, xu, m_w[0][0]);
	afbDtFsD(Xshf.v(), n, xd, m_w[0][1]);
	
	j++;
	
	while(j<nstage && j<(DT_MAX_N_STAGE-1) ) {
	
		const int nj = xu.N();
		
		Xshf.copy(xu, -5);

		afbDtU(Xshf.v(), nj, xu, m_w[j][0]);
			
		Xshf.copy(xd, -5);

		afbDtD(Xshf.v(), nj, xd, m_w[j][1]);
		
		j++;
		
	}
	
	m_w[j][0] = xu;
	m_w[j][1] = xd;
	
	m_numStages = j;
}

void DualTree2::synthesize(VectorN<float> & y)
{
	int j = m_numStages;
	VectorN<float> yu;
	yu.copy(m_w[j][0]);
	VectorN<float> yd;
	yd.copy(m_w[j][1]);

	j--;
	while(j>0) {
		
		sfbDtU(yu, yu, m_w[j][0]);
		sfbDtD(yd, yd, m_w[j][1]);
		
		j--;
	}
	
	sfbDtFsU(yu, yu, m_w[j][0]);
	sfbDtFsD(yd, yd, m_w[j][1]);
	
	const int & n = yu.N();
	y.copy(yu);
	for(int i=0; i<n;++i) {
		y[i] += yd[i];
		y[i] *= 0.707106781187f;
	}
	
}
	
const int & DualTree2::numStages() const
{ return m_numStages; }

const VectorN<float> & DualTree2::stage(const int & i, const int & j ) const
{ return m_w[i][j]; }

}

}