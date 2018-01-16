/*
 *  MACVelocityField.cpp
 *  
 *
 *  Created by jian zhang on 1/17/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "MACVelocityField.h"
#include <math/miscfuncs.h>

namespace aphid {

namespace lbm {

int MACVelocityField::BlockMarkerLength[3] = { 17 * 16 * 16,
16 * 17 * 16,
16 * 16 * 17 };
const int MACVelocityField::ComponentMarkerTable[3][3] = {
{1, 0, 0},
{0, 1, 0},
{0, 0, 1}
};

MACVelocityField::MACVelocityField()
{}

void MACVelocityField::resetBlockVelocity(float* u, float* sum, const int& iblock, const int& d)
{
	m_u[d] = &u[iblock * BlockMarkerLength[d] ]; 
	m_sum[d] = &sum[iblock * BlockMarkerLength[d] ]; 

	memset(m_u[d], 0, BlockMarkerLength[d] * 4);
	memset(m_sum[d], 0, BlockMarkerLength[d] * 4);

}

void MACVelocityField::addVelocity(float* u, float* sum,
			const int& i, const int& j, const int& k,
			const float& xbary, const float& ybary, const float& zbary,
			const float& q,
			const int& d)
{
	float weight;
	int ind;
	
	weight = (1.f - xbary) * (1.f - ybary) * (1.f - zbary);
	ind = MarkerInd(i, j, k, d);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = xbary * (1.f - ybary) * (1.f - zbary);
	ind = MarkerInd(i + 1, j, k, d);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = (1.f - xbary) * ybary * (1.f - zbary);
	ind = MarkerInd(i, j + 1, k, d);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = xbary * ybary * (1.f - zbary);
	ind = MarkerInd(i + 1, j + 1, k, d);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = (1.f - xbary) * (1.f - ybary) * zbary;
	ind = MarkerInd(i, j, k + 1, d);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = xbary * (1.f - ybary) * zbary;
	ind = MarkerInd(i + 1, j, k + 1, d);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = (1.f - xbary) * ybary * zbary;
	ind = MarkerInd(i, j + 1, k + 1, d);
	u[ind] += q * weight;
	sum[ind] += weight;
	
	weight = xbary * ybary * zbary;
	ind = MarkerInd(i + 1, j + 1, k + 1, d);
	u[ind] += q * weight;
	sum[ind] += weight;
	
}

void MACVelocityField::getMarkerCoordWeight(int& i, int& j, int& k,
				float& barx, float& bary, float& barz,
				const float* u, const int& d) const
{
	getCellCoordWeight(i, j, k, barx, bary, barz, u);
	if(d == 0) {
		CenteredCoordWeight(j, bary, 1);
		CenteredCoordWeight(k, barz, 2);
	} else if(d == 1) {
		CenteredCoordWeight(i, barx, 0);
		CenteredCoordWeight(k, barz, 2);
	} else {
		CenteredCoordWeight(i, barx, 0);
		CenteredCoordWeight(j, bary, 1);
	}
}

void MACVelocityField::depositeVelocity(const float* pos, const float* vel)
{
	int i, j, k;
	float xbary, ybary, zbary;
	
	getMarkerCoordWeight(i, j, k, xbary, ybary, zbary, pos, 0);
	
	addVelocity(m_u[0], m_sum[0], i, j, k, xbary, ybary, zbary, vel[0], 0);
	
	getMarkerCoordWeight(i, j, k, xbary, ybary, zbary, pos, 1);
	
	addVelocity(m_u[1], m_sum[1], i, j, k, xbary, ybary, zbary, vel[1], 1);
	
	getMarkerCoordWeight(i, j, k, xbary, ybary, zbary, pos, 2);
	
	addVelocity(m_u[2], m_sum[2], i, j, k, xbary, ybary, zbary, vel[2], 2);
	
}

void MACVelocityField::qdwei(float* q, float* wei, const int& n )
{
	for(int i=0;i<n;++i) {
		if(wei[i] > 0.f)
			q[i] /= wei[i];
	}
}

void MACVelocityField::finishDepositeVelocity()
{
	qdwei(m_u[0], m_sum[0], BlockMarkerLength[0] );
	qdwei(m_u[1], m_sum[1], BlockMarkerLength[1] );
	qdwei(m_u[2], m_sum[2], BlockMarkerLength[2] );
}

int MACVelocityField::MarkerInd(const int& i, const int& j, const int& k, const int& d)
{ 
	if(d == 0 ) {
		return (k * (BlockDim[0] + 1) * BlockDim[1] + j * (BlockDim[0] + 1) + i); 
	}
	if(d == 1) {
		return (k * BlockDim[0] * (BlockDim[1] + 1) + j * BlockDim[0] + i); 
	}
	return (k * BlockDim[0] * BlockDim[1] + j * BlockDim[0] + i); 
}

void MACVelocityField::evaluateCellCenterVelocity(float* u, 
				int& i, int& j, int& k) const
{
	evaluateCellCenterVelocityComponent(u[0], i, j, k, 0);
	evaluateCellCenterVelocityComponent(u[1], i, j, k, 1);
	evaluateCellCenterVelocityComponent(u[2], i, j, k, 2);
}

void MACVelocityField::evaluateCellCenterVelocityComponent(float& u, 
				const int& i, const int& j, const int& k,
				const int& d) const
{
	u = ( m_u[d][MarkerInd(i,     j,     k,     d)]
		+ m_u[d][MarkerInd(i + ComponentMarkerTable[d][0], 
						   j + ComponentMarkerTable[d][1],     
						   k + ComponentMarkerTable[d][2], d)] ) * .5f;
}

void MACVelocityField::evaluateVelocityAtPosition(float* u, const float* p) const
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

void MACVelocityField::extractCellVelocities(float* u) const
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

void MACVelocityField::clearVelocities()
{
	for(int i=0;i<3;++i) {
		memset(m_u[i], 0, BlockMarkerLength[i] * 4);
		memset(m_sum[i], 0, BlockMarkerLength[i] * 4);
	}
}

void MACVelocityField::depositeCellCenterVelocity(const int& i, const int& j, const int& k,
				const float* vel)
{
	int ind;

	for(int d=0;d<3;++d) {
		ind = MarkerInd(i, j, k, d);
		m_u[d][ind] += vel[d] * .5f;
		m_sum[d][ind] += .5f;
	
		ind = MarkerInd(i + ComponentMarkerTable[d][0], 
						j + ComponentMarkerTable[d][1], 
						k + ComponentMarkerTable[d][2], d);
						
		m_u[d][ind] += vel[d] * .5f;
		m_sum[d][ind] += .5f;
	}
}

}

}