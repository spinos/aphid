/*
 *  LatticeBlock.cpp
 *
 *  Created by jian zhang on 1/16/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "LatticeBlock.h"
#include "D3Q19DF.h"
#include <math/linearMath.h>
#include <math/miscfuncs.h>
#include <boost/thread.hpp>

namespace aphid {

namespace lbm {

float* LatticeBlock::RestQ[19];

LatticeBlock::LatticeBlock(sdb::Entity * parent) : sdb::Entity(parent)
{}

void LatticeBlock::resetDensity(float* rho, const int& iblock)
{ m_rho = &rho[iblock * BlockLength]; }

void LatticeBlock::resetFlag(char* fg, const int& iblock)
{ m_flag = &fg[iblock * BlockLength]; }

void LatticeBlock::resetQi(float* q, const int& iblock, const int& i)
{ m_q[i] = &q[iblock * BlockLength]; }

void LatticeBlock::simulationStep()
{
	if(numParticlesInGrid() < 1)
		return;
		
/// no stream on q_0
	for(int i=1;i<19;++i) {
		streaming(i);
	}
/// boundary conditions here
	collision();
}

void LatticeBlock::streaming(const int& i)
{
	float* q_i = m_q[i];
	float* tmp = m_q[19];
	memcpy(tmp, q_i, BlockLength * 4);
	
	int c_i[3];
	D3Q19DF::GetStreamDirection(c_i, i);
	
#if 1
	boost::thread strmThread[8];
	for(int i=0;i<8;++i) {
		strmThread[i] = boost::thread( boost::bind(D3Q19DF::ComputeStreaming,
								q_i, tmp, c_i, ZRank8Begins[i], ZRank8Begins[i+1], BlockDim) );
	}
	
	for(int i=0;i<8;++i) {
		strmThread[i].join();
	}
#else
	for(int k=0; k<BlockDim[2];++k) {
		for(int j=0;j<BlockDim[1];++j) {
			for(int i=0;i<BlockDim[0];++i) {
				if(IsCellCoordValid(i + c_i[0], j + c_i[1], k + c_i[2] ) ) 
					q_i[CellInd(i, j, k)] = tmp[CellInd(i + c_i[0], j + c_i[1], k + c_i[2])];
			}
		}
	}
#endif
}

void LatticeBlock::collision()
{
#if 1
	const float omega = 1.9f;
	boost::thread collThread[8];
	for(int i=0;i<8;++i) {
		collThread[i] = boost::thread( boost::bind(D3Q19DF::ComputeCollision,
								m_q, omega, ZInd8Begins[i], ZInd8Begins[i+1]) );
	}
	
	for(int i=0;i<8;++i) {
		collThread[i].join();
	}
#else	
	float u[3];
	float rho, uu, omega = 1.9f;
	int ii, j, k;
	for(int i=0;i<BlockLength;++i) {
		
		D3Q19DF::IncompressibleVelocity(u, rho, m_q, i);
		
		uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
		
		D3Q19DF::Relaxing(m_q, u, uu, rho, omega, i);
	}
#endif
}

void LatticeBlock::rankedInitialCondition(const int& zbegin, const int& zend)
{
	float u[3], uu;
	int ind;
	for(int k=zbegin; k<zend;++k) {
		for(int j=0;j<BlockDim[1];++j) {
			for(int i=0;i<BlockDim[0];++i) {
				evaluateCellCenterVelocity(u, i, j, k);
				uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
				
				if(uu < 1e-8f)
					continue;
					
				ind = CellInd(i, j, k);
				m_flag[ind] = 1;
				D3Q19DF::DiscretizeVelocity(m_q, 1.f, u, ind );

			}
		}
	}
}

void LatticeBlock::initialCondition()
{
	if(numParticlesInGrid() < 1)
		return;
		
	memset(m_flag, 0, BlockLength);
	for(int i=0;i<19;++i) {
		memcpy(m_q[i], RestQ[i], BlockLength<<2);
	}
	
	jacobi(.7f, .08f);
#if 1
	boost::thread initThread[8];
	for(int i=0;i<8;++i) {
		initThread[i] = boost::thread( boost::bind(&LatticeBlock::rankedInitialCondition, this,
										ZRank8Begins[i], ZRank8Begins[i+1]) );
	}
	
	for(int i=0;i<8;++i) {
		initThread[i].join();
	}
#else
	float u[3], uu;
	for(int k=0; k<BlockDim[2];++k) {
		for(int j=0;j<BlockDim[1];++j) {
			for(int i=0;i<BlockDim[0];++i) {
				evaluateCellCenterVelocity(u, i, j, k);
				uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
				
				if(uu < 1e-8f)
					continue;
					
				m_flag[CellInd(i, j, k)] = 1;
				D3Q19DF::DiscretizeVelocity(m_q, 1.f, u, CellInd(i, j, k) );

			}
		}
	}
#endif
}

void LatticeBlock::boundaryCondition(const float& bcRho)
{
	float u[3];
	int ii, j, k;
	for(int i=0;i<BlockLength;++i) {
		if(m_flag[i] > 0) {
			CellCoord(ii, j, k, i);
			evaluateCellCenterVelocity(u, ii, j, k);
			D3Q19DF::IncomingBC(m_q, u, bcRho, i);
		}
	}
}

void LatticeBlock::updateVelocityDensity()
{
	if(numParticlesInGrid() < 1)
		return;
		
	clearVelocities();

	float u[3];
	float uu;
	for(int k=0; k<BlockDim[2];++k) {
		for(int j=0;j<BlockDim[1];++j) {
			for(int i=0;i<BlockDim[0];++i) {
				
				const int ind = CellInd(i, j, k);
				D3Q19DF::IncompressibleVelocity(u, m_rho[ind], m_q, ind );
				uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
				
				if(uu < 1e-9f) 
					continue;
					
				depositeCellCenterVelocity(i, j, k, u);

			}
		}
	}
}

void LatticeBlock::limitSpeed(float& x) const
{ ClampInPlace<float>(x, -.3f, .3f); }

void LatticeBlock::extractCellDensities(float* dst)
{ 
	for(int i=0;i<BlockLength;++i) {
		D3Q19DF::Density(dst[i], m_q, i );
	}
}

void LatticeBlock::evaluateVelocityDensityAtPosition(float* u, float& rho, const float* p)
{
	evaluateVelocityAtPosition(u, p);
	rho = m_rho[getCellInd(p) ];
}

void LatticeBlock::BuildRestQ()
{
	for(int i=0;i<19;++i) {
		RestQ[i] = new float[BlockLength];
		D3Q19DF::SetWi(RestQ[i], BlockLength, i);
	}
}

void LatticeBlock::modifyParticleVelocities(float* vel, const float* pos,
					const int& np, const float& scaling)
{
	if(numParticlesInGrid() < 1)
		return;
		
	int nppt = np>>3;
	if(nppt < 1)
		nppt = 1;
	int indpt[9];
	for(int i=0;i<8;++i) {
		indpt[i] = nppt * i;
	}
	indpt[8] = np;
	
	boost::thread modvThread[8];
	for(int i=0;i<8;++i) {
		modvThread[i] = boost::thread( boost::bind(&LatticeBlock::blockModifyParticleVelocities, this,
								vel, pos, scaling, indpt[i], indpt[i+1]) );
	}
	
	for(int i=0;i<8;++i) {
		modvThread[i].join();
	}
}

void LatticeBlock::blockModifyParticleVelocities(float* vel, const float* pos, const float& scaling,
					const int& ibegin, const int& iend)
{
	float vout[3], rho, weight;
	for(int i=ibegin;i<iend;++i) {
		const float* posi = &pos[i * 3];
		if(isPointOutsideBound(posi))
			continue;
			
		evaluateVelocityDensityAtPosition(vout, rho, posi);
		
		weight = rho - 1.f;
		if(weight < 0.f)
			continue;
			
		weight /= .13f;
			
		if(weight > 1.f)
			weight = 1.f;
			
		float* veli = &vel[i * 3];

		vout[0] *= scaling;
		vout[1] *= scaling;
		vout[2] *= scaling;
		
		veli[0] += (vout[0] - veli[0]) * weight;
		veli[1] += (vout[1] - veli[1]) * weight;
		veli[2] += (vout[2] - veli[2]) * weight;
	}
}

}

}
