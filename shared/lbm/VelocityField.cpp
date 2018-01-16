/*
 *  VelocityField.cpp
 *  
 *
 *  Created by jian zhang on 1/17/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "VelocityField.h"
#include <math/miscfuncs.h>

namespace aphid {

namespace lbm {

VelocityField::VelocityField()
{}

void VelocityField::resetBlockVelocity(float* u, float* sum, const int& iblock, const int& d)
{
	m_u[d] = &u[iblock * BlockLength ]; 
	m_sum[d] = &sum[iblock * BlockLength ]; 

	memset(m_u[d], 0, BlockLength * 4);
	memset(m_sum[d], 0, BlockLength * 4);

}

void VelocityField::addVelocity(float* u, float* sum,
			const int& i, const int& j, const int& k,
			const float& xbary, const float& ybary, const float& zbary,
			const float& q,
			const int& d)
{
	float weight;
	int ind;
	
	weight = (1.f - xbary) * (1.f - ybary) * (1.f - zbary);
	ind = CellInd(i, j, k);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = xbary * (1.f - ybary) * (1.f - zbary);
	ind = CellInd(i + 1, j, k);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = (1.f - xbary) * ybary * (1.f - zbary);
	ind = CellInd(i, j + 1, k);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = xbary * ybary * (1.f - zbary);
	ind = CellInd(i + 1, j + 1, k);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = (1.f - xbary) * (1.f - ybary) * zbary;
	ind = CellInd(i, j, k + 1);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = xbary * (1.f - ybary) * zbary;
	ind = CellInd(i + 1, j, k + 1);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = (1.f - xbary) * ybary * zbary;
	ind = CellInd(i, j + 1, k + 1);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = xbary * ybary * zbary;
	ind = CellInd(i + 1, j + 1, k + 1);
	u[ind] += q * weight;
	sum[ind] += weight;
	
}

void VelocityField::depositeVelocity(const float* pos, const float* vel)
{
	int i, j, k;
	float xbary, ybary, zbary;
	
	getCellCoordWeight(i, j, k, xbary, ybary, zbary, pos);
	CenteredCoordWeight(i, xbary, 0);
	CenteredCoordWeight(j, ybary, 1);
	CenteredCoordWeight(k, zbary, 2);
	
	addVelocity(m_u[0], m_sum[0], i, j, k, xbary, ybary, zbary, vel[0], 0);
	addVelocity(m_u[1], m_sum[1], i, j, k, xbary, ybary, zbary, vel[1], 1);
	addVelocity(m_u[2], m_sum[2], i, j, k, xbary, ybary, zbary, vel[2], 2);
	
}

void VelocityField::qdwei(float* q, float* wei, const int& n )
{
	for(int i=0;i<n;++i) {
		if(wei[i] > 0.f)
			q[i] /= wei[i];
	}
}

void VelocityField::finishDepositeVelocity()
{
	qdwei(m_u[0], m_sum[0], BlockLength );
	qdwei(m_u[1], m_sum[1], BlockLength );
	qdwei(m_u[2], m_sum[2], BlockLength );
}

void VelocityField::evaluateCellCenterVelocity(float* u, 
				int& i, int& j, int& k) const
{
	int ind = CellInd(i, j, k);
	u[0] = m_u[0][ind];
	u[1] = m_u[1][ind];
	u[2] = m_u[2][ind];
}

void VelocityField::evaluateCellCenterVelocityComponent(float& u, 
				const int& i, const int& j, const int& k,
				const int& d) const
{ u = m_u[d][CellInd(i, j, k)]; }

void VelocityField::evaluateVelocityAtPosition(float* u, const float* p) const
{
	int i, j, k;
	float barx, bary, barz;
	getCellCoordWeight(i, j, k, barx, bary, barz, p);
	CenteredCoordWeight(i, barx, 0);
	CenteredCoordWeight(j, bary, 1);
	CenteredCoordWeight(k, barz, 2);
	float c[2][4];
	
	for(int d=0;d<3;++d) {
/// 8 corners
		for(int zi = 0;zi<2;++zi) {
			evaluateCellCenterVelocityComponent(c[zi][0], i    , j    , k + zi, d);
			evaluateCellCenterVelocityComponent(c[zi][1], i + 1, j    , k + zi, d);
			evaluateCellCenterVelocityComponent(c[zi][2], i    , j + 1, k + zi, d);
			evaluateCellCenterVelocityComponent(c[zi][3], i + 1, j + 1, k + zi, d);
		}
		TrilinearInterpolation(u[d], c, barx, bary, barz);
	}
	
}

void VelocityField::extractCellVelocities(float* u) const
{
	for(int k=0; k<BlockDim[2];++k) {
		for(int j=0;j<BlockDim[1];++j) {
			for(int i=0;i<BlockDim[0];++i) {
				float* v = &u[CellInd(i, j, k) * 3];
				evaluateCellCenterVelocity(v, i, j, k);
			}
		}
	}
}

void VelocityField::clearVelocities()
{
	for(int i=0;i<3;++i) {
		memset(m_u[i], 0, BlockLength * 4);
		memset(m_sum[i], 0, BlockLength * 4);
	}
}

void VelocityField::depositeCellCenterVelocity(const int& i, const int& j, const int& k,
				const float* vel)
{
	int ind;

	for(int d=0;d<3;++d) {
		ind = CellInd(i, j, k);
		m_u[d][ind] += vel[d];
		m_sum[d][ind] += 1.f;
	}
}

}

}