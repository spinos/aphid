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
	LatticeBlock::BuildRestQ();
	m_rebuildGridFlag = true;
}

LatticeManager::~LatticeManager()
{}

void LatticeManager::setParam(const LatticeParam& param)
{
	m_param._blockSize = param._blockSize > 16.f ? param._blockSize : 16.f;
	m_param._inScale = param._inScale;
	m_param._outScale = param._outScale;
	resetLattice();
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
	
	m_rho.reset(new DenseVector<float>(LatticeBlock::BlockLength * 16 ) );
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

void LatticeManager::buildLattice(const float* p,
					const int& np)
{
	sdb::Sequence<sdb::Coord3 > allCoords;
	for(int i=0;i<np;++i) {
		const float* pos = &p[i * 3];
		const sdb::Coord3 c = m_grid.gridCoord(pos);
		allCoords.insert(c);
	}
	
	allCoords.begin();
	while(!allCoords.end() ) {
	
		m_grid.insertCell(allCoords.key() );
		
		allCoords.next();
	}
	
	const float& blockSize = m_grid.gridSize();
	
	m_numBlocks = 0;
	
	m_grid.begin();
	while(!m_grid.end() ) {
		
		const sdb::Coord3 c = m_grid.key();
		
		if((m_numBlocks + 1) > m_capBlocks) {
			extendArrays();
		}
		
		resetBlock(m_grid.value(), blockSize * c.x,
							blockSize * c.y,
							blockSize * c.z);
		
		m_numBlocks++;
				
		m_grid.next();
	}
}

void LatticeManager::injectParticles(const float* p,
					const float* v,
					const int& np)
{
	if(m_rebuildGridFlag)
		buildLattice(p, np);
	
	const float scaling = LatticeBlock::OneOverH * m_param._inScale;	
	int nadded = 0;
	m_grid.begin();
	while(!m_grid.end() ) {
		m_grid.value()->depositeVelocities(nadded, v, p, np, scaling);
		
		m_grid.next();
	}
	
	m_rebuildGridFlag = (nadded < np);
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
	
	m_rho->expand(LatticeBlock::BlockLength * 16 );
	
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
	
	blk->resetDensity(m_rho->v(), m_numBlocks);
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
		m_grid.value()->updateVelocityDensity();
		
		m_grid.next();
	}
}

void LatticeManager::modifyParticleVelocities(float* vel,
					const float* pos,
					const int& np)
{
	const float scaling = LatticeBlock::CellSize * m_param._outScale;

	m_grid.begin();
	while(!m_grid.end() ) {
		m_grid.value()->modifyParticleVelocities(vel, pos, np, scaling);
		
		m_grid.next();
	}
}

}

}