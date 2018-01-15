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
{}

LatticeManager::~LatticeManager()
{}

void LatticeManager::resetLattice(const LatticeParam& param)
{
	for(int i=0;i<19;++i) {
		m_q[i].reset(new DenseVector<float>(65536) );
	}
	
	m_grid.clear();
	m_grid.setGridSize(param._blockSize);
	LatticeBlock::NodeSize = param._blockSize / 16.f;
	LatticeBlock::HalfNodeSize = 0.5f * LatticeBlock::NodeSize;
	LatticeBlock::OneOverH = 1.f / LatticeBlock::NodeSize;
	m_numBlocks = 0;
	m_capBlocks = 16;
}

sdb::WorldGrid2<LatticeBlock >& LatticeManager::grid()
{ return m_grid; }

void LatticeManager::injectParticles(const float* p,
					const float* v,
					const int& np)
{
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
				curCell->setCorner(blockSize * c.x,
									blockSize * c.y,
									blockSize * c.z);
				curCell->setQOffset(m_numBlocks * LatticeBlock::BlockLength);
				
				initializeBlockQ(curCell->qOffset() );
				
				if((m_numBlocks + 1) > m_capBlocks) {
					extendQ();
				}
				
				m_numBlocks++;
			}
			prevCoord = c;
			prevCell = curCell;
			
		}
		
		curCell->calcNodeCoord(cu, bu, pos[0], 0);
		curCell->calcNodeCoord(cv, bv, pos[1], 1);
		curCell->calcNodeCoord(cw, bw, pos[2], 2);

		const float* vel = &v[i * 3];
		
		addVelocity(cu, cv, cw, bu, bv, bw, vel, curCell->qOffset() );
		
	}
}

void LatticeManager::extendQ()
{
	for(int i=0;i<19;++i) {
		m_q[i]->expand(65536);
	}
	m_capBlocks += 16;
	
}

void LatticeManager::initializeBlockQ(const int& begin)
{
	for(int i=0;i<19;++i) {
		LatticeBlock::InitializeQ(&m_q[i]->v()[begin], i );
	}
}

void LatticeManager::addVelocity(const int& u, const int& v, const int& w,
				const float& bu, const float& bv, const float& bw,
				const float* vel,
				const int& begin)
{
	const float ub = 1.f - bu;
	const float vb = 1.f - bv;
	const float wb = 1.f - bw;
	float vv[3];
	float wei = ub * vb * wb;
	
	vv[0] = vel[0] * wei;
	vv[1] = vel[1] * wei;
	vv[2] = vel[2] * wei;
	
	for(int i=0;i<19;++i) {
		LatticeBlock::AddQ(u, v, w, vv, &m_q[i]->v()[begin], i );
	}
	
	wei = bu * vb * wb;
	
	vv[0] = vel[0] * wei;
	vv[1] = vel[1] * wei;
	vv[2] = vel[2] * wei;
	
	for(int i=0;i<19;++i) {
		LatticeBlock::AddQ(u + 1, v, w, vv, &m_q[i]->v()[begin], i );
	}
	
	wei = ub * bv * wb;
	
	vv[0] = vel[0] * wei;
	vv[1] = vel[1] * wei;
	vv[2] = vel[2] * wei;
	
	for(int i=0;i<19;++i) {
		LatticeBlock::AddQ(u, v + 1, w, vv, &m_q[i]->v()[begin], i );
	}
	
	wei = bu * bv * wb;
	
	vv[0] = vel[0] * wei;
	vv[1] = vel[1] * wei;
	vv[2] = vel[2] * wei;
	
	for(int i=0;i<19;++i) {
		LatticeBlock::AddQ(u + 1, v + 1, w, vv, &m_q[i]->v()[begin], i );
	}
	
	wei = ub * vb * bw;
	
	vv[0] = vel[0] * wei;
	vv[1] = vel[1] * wei;
	vv[2] = vel[2] * wei;
	
	for(int i=0;i<19;++i) {
		LatticeBlock::AddQ(u, v, w + 1, vv, &m_q[i]->v()[begin], i );
	}
	
	wei = bu * vb * bw;
	
	vv[0] = vel[0] * wei;
	vv[1] = vel[1] * wei;
	vv[2] = vel[2] * wei;
	
	for(int i=0;i<19;++i) {
		LatticeBlock::AddQ(u + 1, v, w + 1, vv, &m_q[i]->v()[begin], i );
	}
	
	wei = ub * bv * bw;
	
	vv[0] = vel[0] * wei;
	vv[1] = vel[1] * wei;
	vv[2] = vel[2] * wei;
	
	for(int i=0;i<19;++i) {
		LatticeBlock::AddQ(u, v + 1, w + 1, vv, &m_q[i]->v()[begin], i );
	}
	
	wei = bu * bv * bw;
	
	vv[0] = vel[0] * wei;
	vv[1] = vel[1] * wei;
	vv[2] = vel[2] * wei;
	
	for(int i=0;i<19;++i) {
		LatticeBlock::AddQ(u + 1, v + 1, w + 1, vv, &m_q[i]->v()[begin], i );
	}
}

}

}