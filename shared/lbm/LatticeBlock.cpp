/*
 *  LatticeBlock.cpp
 *  
 *
 *  Created by jian zhang on 1/16/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "LatticeBlock.h"
#include <math/miscfuncs.h>

namespace aphid {

namespace lbm {

// the weight for the equilibrium distribution w
static const float w_alpha[19] = { (1./3.), 
	(1./18.),(1./18.),(1./18.),(1./18.),(1./18.),(1./18.),
	(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),
	(1./36.),(1./36.),(1./36.),(1./36.),(1./36.),(1./36.), };
	
// the discrete set of microscopic velocities e
static const int e_x[19] = { 0, 0,0,  1,-1, 0,0,  1,-1,1,-1, 0,0,0,0,   1,1,-1,-1 };
static const int e_y[19] = { 0, 1,-1, 0,0,  0,0,  1,1,-1,-1, 1,1,-1,-1, 0,0,0,0 };
static const int e_z[19] = { 0, 0,0,  0,0,  1,-1, 0,0,0,0,   1,-1,1,-1, 1,-1,1,-1 };

float LatticeBlock::NodeSize = 1.f;
float LatticeBlock::HalfNodeSize = .5f;
float LatticeBlock::OneOverH = 1.f / 16.f;
int LatticeBlock::BlockLength = 4096;

LatticeBlock::LatticeBlock(sdb::Entity * parent) : sdb::Entity(parent)
{}

void LatticeBlock::setCorner(const float& x, const float& y, const float& z)
{ 
	m_corner[0] = x;
	m_corner[1] = y;
	m_corner[2] = z;
}

void LatticeBlock::setQOffset(const int& x)
{ m_qoffset = x; }

const int& LatticeBlock::qOffset() const
{ return m_qoffset; }

void LatticeBlock::calcNodeCoord(int& i, float& bary, const float& u, 
					const int& d) const
{
	const float lu = (u - m_corner[d]) * OneOverH - .5f;
	i = lu;
	bary = lu - floor(lu);
}

void LatticeBlock::InitializeQ(float* q, const int& i)
{
	for(int j=0;j<BlockLength;++j) {
		q[j] = w_alpha[i];
	}
}

void LatticeBlock::AddQ(const int& u, const int& v, const int& w,
				const float* vel,
				float* q, const int& i)
{
	if(IsNodeIndOutOfBound(u, v, w) )
		return;
			
	q[NodeInd(u, v, w)] += (e_x[i] * vel[0] + e_y[i] * vel[1] + e_z[i] * vel[2]) * w_alpha[i];
}

bool LatticeBlock::IsNodeIndOutOfBound(const int& i, const int& j, const int& k)
{
	if(i < 0 || i > 15)
		return true;
		
	if(j < 0 || j > 15)
		return true;
		
	if(k < 0 || k > 15)
		return true;
		
	return false;
}

int LatticeBlock::NodeInd(const int& i, const int& j, const int& k)
{ return ((k<<8) + (j<<4) + i); }

}

}
