/*
 *  dtdwt1.cpp
 *  
 *
 *  Created by jian zhang on 9/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "dtdwt1.h"
#include <iostream>

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
	float * Xshf = wla::circshift(x.v(), n, -5);
/// normalize
	int i=0;
	for(;i<n;++i) 
		Xshf[i] *= 0.707106781187f;
	
	VectorN<float> x1;
	x1.create(n);
	VectorN<float> x2;
	x2.create(n);
	
	int j = 0; 
	
	afbDtFsU(Xshf, n, x1, m_w[0][0]);
	afbDtFsD(Xshf, n, x2, m_w[0][1]);
	
	delete[] Xshf;
	
	j++;
	
	while(j<nstage && j<(DT_MAX_N_STAGE-1) ) {
	
		const int nj = x1.N();
		
		afbDtU(x1.v(), nj, x1, m_w[j][0]);			
		afbDtD(x2.v(), nj, x2, m_w[j][1]);
		
		j++;
		
	}
	
	m_w[j][0] = x1;
	m_w[j][1] = x2;
	
	m_numStages = j;
}

void DualTree::synthesize(VectorN<float> & y)
{
}
	
const int & DualTree::numStages() const
{ return m_numStages; }

const VectorN<float> & DualTree::stage(const int & i, const int & j ) const
{ return m_w[i][j]; }

}

}