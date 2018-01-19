/*
 *  LatticeManager.cpp
 *  
 *
 *  Created by jian zhang on 1/15/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "LatticeManager.h"
#include "LatticeBlock.h"

namespace aphid {

namespace lbm {

LatticeManager::LatticeManager()
{
	m_param._blockSize = 16.f; 
	m_param._inScale = 1.f;
	m_param._outScale = 1.f;
}

LatticeManager::~LatticeManager()
{}

void LatticeManager::setParam(const LatticeParam& param)
{
	m_param._blockSize = param._blockSize > 16.f ? param._blockSize : 16.f;
	m_param._inScale = param._inScale;
	m_param._outScale = param._outScale;
}

void LatticeManager::resetLattice()
{
	for(int i=0;i<20;++i) {
		m_q[i].reset(new DenseVector<float>(LatticeBlock::BlockLength * 16 ) );
	}
	
	for(int i=0;i<3;++i) {
		m_u[i].reset(new DenseVector<float>(LatticeBlock::BlockMarkerLength[i] * 16 ) );
		m_sum[i].reset(new DenseVector<float>(LatticeBlock::BlockMarkerLength[i] * 16 ) );
	}
	
	m_flag.reset(new char[LatticeBlock::BlockLength * 16]);
	
	m_grid.clear();
	m_grid.setGridSize(m_param._blockSize);
	LatticeBlock::CellSize = m_grid.gridSize() / (float)LatticeBlock::BlockDim[0];
	LatticeBlock::HalfCellSize = 0.5f * LatticeBlock::CellSize;
	LatticeBlock::OneOverH = 1.f / LatticeBlock::CellSize;
	m_numBlocks = 0;
	m_capBlocks = 16;
}

sdb::WorldGrid2<LatticeBlock >& LatticeManager::grid()
{ return m_grid; }

DenseVector<float>& LatticeManager::q_i(const int& i)
{ return *m_q[i]; }

void LatticeManager::injectParticles(const float* p,
					const float* v,
					const int& np)
{
	const float scaling = LatticeBlock::OneOverH * m_param._inScale;
	
	LatticeBlock* prevCell = 0;
	LatticeBlock* curCell;
/// unlikely
	sdb::Coord3 prevCoord(999999,999999,999999);
	int cu, cv, cw;
	float bu, bv, bw;
	const float& blockSize = m_grid.gridSize();
	for(int i=0;i<np;++i) {
		const float* pos = &p[i * 3];
		const sdb::Coord3 c = m_grid.gridCoord(pos);
		
		if(c == prevCoord) {
			curCell = prevCell;
			
		} else {
			curCell = m_grid.findCell(c);
			if(!curCell) {
				curCell = m_grid.insertCell(c);
				
				if((m_numBlocks + 1) > m_capBlocks) {
					extendArrays();
				}
				
				resetBlock(curCell, blockSize * c.x,
									blockSize * c.y,
									blockSize * c.z);
				
				m_numBlocks++;
			}
			prevCoord = c;
			prevCell = curCell;
			
		}
		
		const float* vel = &v[i * 3];
		
		curCell->depositeVelocity(pos, vel, scaling );
		
	}
}

void LatticeManager::finishInjectingParticles()
{
	m_grid.begin();
	while(!m_grid.end() ) {
		m_grid.value()->finishDepositeVelocity();
		
		m_grid.next();
	}
}

void LatticeManager::extendArrays()
{
	for(int i=0;i<20;++i) {
		m_q[i]->expand(LatticeBlock::BlockLength * 16 );
	}
	
	for(int i=0;i<3;++i) {
		m_u[i]->expand(LatticeBlock::BlockMarkerLength[i] * 16 );
		m_sum[i]->expand(LatticeBlock::BlockMarkerLength[i] * 16 );
	}
	
	char* tmp = new char[LatticeBlock::BlockLength * m_capBlocks];
	memcpy(tmp, m_flag.get(), LatticeBlock::BlockLength * m_capBlocks );
	
	m_flag.reset(new char[LatticeBlock::BlockLength * (m_capBlocks + 16) ]);
	memcpy(m_flag.get(), tmp, LatticeBlock::BlockLength * m_capBlocks );
	
	delete[] tmp;
	
	m_capBlocks += 16;
	
}

void LatticeManager::resetBlock(LatticeBlock* blk, const float& cx, const float& cy, const float& cz)
{
	blk->setCorner(cx, cy, cz);
				
	blk->resetBlockVelocity(m_u[0]->v(), m_sum[0]->v(),
							m_numBlocks, 0);
	blk->resetBlockVelocity(m_u[1]->v(), m_sum[1]->v(),
							m_numBlocks, 1);
	blk->resetBlockVelocity(m_u[2]->v(), m_sum[2]->v(),
							m_numBlocks, 2);
							
	for(int i=0;i<20;++i) {
		blk->resetQi(m_q[i]->v(), m_numBlocks, i );
	}
	
	blk->resetFlag(m_flag.get(), m_numBlocks);
}

void LatticeManager::initialCondition()
{
	m_grid.begin();
	while(!m_grid.end() ) {
		m_grid.value()->initialCondition();
		
		m_grid.next();
	}
}

void LatticeManager::simulationStep()
{
	m_grid.begin();
	while(!m_grid.end() ) {
		m_grid.value()->simulationStep();
		m_grid.value()->updateVelocity();
		
		m_grid.next();
	}
}

void LatticeManager::modifyParticleVelocities(float* vel,
					const float* pos,
					const int& np)
{
	const float scaling = LatticeBlock::CellSize * m_param._outScale;
	
	LatticeBlock* prevCell = 0;
	LatticeBlock* curCell;
/// unlikely
	sdb::Coord3 prevCoord(999999,999999,999999);
	int cu, cv, cw;
	float bu, bv, bw, vout[3], rho;
	
	for(int i=0;i<np;++i) {
		const float* posi = &pos[i * 3];
		const sdb::Coord3 c = m_grid.gridCoord(posi);
		
		if(c == prevCoord) {
			curCell = prevCell;
			
		} else {
			curCell = m_grid.findCell(c);
			if(curCell) {
				prevCoord = c;
				prevCell = curCell;
			}
		}
		
		if(!curCell) {
			std::cout<<"\n LatticeManager::modifyParticleVelocities cannot find cell at coord"<<c;
			continue;
		}
		
		curCell->evaluateVelocityDensityAtPosition(vout, rho, posi);
		
		float weight = rho - 1.f;
		if(weight < 0.f)
			continue;
			
		weight /= .09f;
			
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