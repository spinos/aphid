/*
 *  UniformGrid.cpp
 *  
 *
 *  Created by jian zhang on 1/19/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "UniformGrid.h"

#include <math/miscfuncs.h>

namespace aphid {

namespace lbm {

int UniformGrid::BlockDim[3] = {16, 16, 16};
int UniformGrid::BlockLength = 4096;
float UniformGrid::CellSize = 1.f;
float UniformGrid::HalfCellSize = .5f;
float UniformGrid::OneOverH = 1.f / 16.f;

UniformGrid::UniformGrid()
{}

void UniformGrid::setCorner(const float& x, const float& y, const float& z)
{ 
	m_corner[0] = x;
	m_corner[1] = y;
	m_corner[2] = z;
}

void UniformGrid::CenteredCoordWeight(int& i, float& bary, const int& d)
{
	bary -= .5f;
	if(bary < 0.f) {
		bary += 1.f;
		i--;
		if(i < 0) {
			i = 0;
			bary = 0.f;
		}
	} else {
		if(i >= BlockDim[d] - 1) {
			i = BlockDim[d] - 2;
			bary = 1.f;
		}
	}
}

void UniformGrid::getCellCoordWeight(int& i, int& j, int& k,
				float& barx, float& bary, float& barz,
				const float* u) const
{
	float lu = (u[0] - m_corner[0]) * OneOverH;
	float lv = (u[1] - m_corner[1]) * OneOverH;
	float lw = (u[2] - m_corner[2]) * OneOverH;
	i = lu;
	j = lv;
	k = lw;
	barx = lu - i;
	bary = lv - j;
	barz = lw - k;
}

int UniformGrid::CellInd(const int& i, const int& j, const int& k)
{ return (k * BlockDim[0] * BlockDim[1] + j * BlockDim[0] + i); }

void UniformGrid::CellCoord(int& i, int& j, int& k, const int& ind)
{
	k = ind / (BlockDim[0] * BlockDim[1]);
	j = (ind - k * (BlockDim[0] * BlockDim[1]) ) / BlockDim[0];
	i = ind - k * (BlockDim[0] * BlockDim[1]) - j * BlockDim[0];
}

void UniformGrid::TrilinearInterpolation(float& u, float c[][4], 
				const float& barx, const float& bary, const float& barz )
{
	c[0][0] = c[0][0] * (1.f - barx) + c[0][1] * barx;
	c[0][1] = c[0][2] * (1.f - barx) + c[0][3] * barx;
	c[1][0] = c[1][0] * (1.f - barx) + c[1][1] * barx;
	c[1][1] = c[1][2] * (1.f - barx) + c[1][3] * barx;
	
	c[0][0] = c[0][0] * (1.f - bary) + c[0][1] * bary;
	c[1][0] = c[1][0] * (1.f - bary) + c[1][1] * bary;
	
	u = c[0][0] * (1.f - barz) + c[1][0] * barz;
}

void UniformGrid::extractCellCenters(float* p) const
{
	for(int k=0; k<BlockDim[2];++k) {
		for(int j=0;j<BlockDim[1];++j) {
			for(int i=0;i<BlockDim[0];++i) {
				float* v = &p[CellInd(i, j, k) * 3];
				
				v[0] = m_corner[0] + CellSize * i + HalfCellSize;
				v[1] = m_corner[1] + CellSize * j + HalfCellSize;
				v[2] = m_corner[2] + CellSize * k + HalfCellSize;
			}
		}
	}
}

}

}