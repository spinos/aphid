/*
 *  SuperformulaPoisson.cpp
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "SuperformulaPoisson.h"
#include <iostream>

using namespace aphid;
namespace ttg {

SuperformulaPoisson::SuperformulaPoisson()
{}

SuperformulaPoisson::~SuperformulaPoisson() 
{}

const char * SuperformulaPoisson::titleStr() const
{ return "Superformula / Poisson-disk Sampling"; }

bool SuperformulaPoisson::createSamples()
{
	BoundingBox box;
	int i, j;
	const float du = 2.f * 3.14159269f / 120.f;
	const float dv = 3.14159269f / 60.f;
	
	for(j=0;j<60;++j) {
		for(i=0;i<120;++i) {
			box.expandBy(randomPnt(du * i - 3.1415927f, dv * j - 1.507963f) );
		}
	}

#define GFACTOR 10.f
//#define RFACTOR .0909f /// 1 / 11
//#define RFACTOR .0625f /// 1 / 16
//#define RFACTOR .03125f /// 1 / 32
#define RFACTOR .01 /// 1 / 100
#define NFACTOR 20000
#define LFACTOR 400000
	float r = box.getLongestDistance() * RFACTOR;
	std::cout<<"\n r "<<r;
	float gridSize = r * GFACTOR;
	
	m_bkg.clear();
	m_bkg.setGridSize(gridSize);
	
	setN(NFACTOR);
	
	int numAccept = 0, preN = 0;
	Disk cand;
	cand.r = r * .5f;

	for(i=0; i<LFACTOR; ++i) {
	
		cand.pos = randomPnt(RandomFn11() * 3.14159269f, 
								RandomFn11() * 3.14159269f * .5f);
		
		if(!m_bkg.reject(&cand) ) {
			
			Disk * samp = new Disk;
			samp->pos = cand.pos;
			samp->r = cand.r;
			samp->key = numAccept;
			
			m_bkg.insert((const float *)&cand.pos, samp);
			
			X()[numAccept++] = cand.pos;
		}
		
		if((i+1) & 255 == 0) {
			if(numAccept == preN) break;
			preN = numAccept;
		}
		if(numAccept >= NFACTOR) break;
	}
	m_bkg.calculateBBox();
	
	setNDraw(numAccept);
	
	std::cout<<"\n n accept "<<numAccept;
	
	std::cout<<"\n sample grid size "<<m_bkg.gridSize()
		<<"\n n cell "<<m_bkg.size()
		<<"\n bbx "<<m_bkg.boundingBox();
	
	std::cout.flush();
	return true;
}

void SuperformulaPoisson::fillBackgroud(PoissonSequence<Disk> * dst,
						PoissonSequence<Disk> * frontGrid,
						int & n)
{
	const float frontSize = frontGrid->gridSize();
	Vector3F center;
	sdb::Coord3 ccenter;
	Disk cand;
	cand.r = frontSize * .499f;
	
	frontGrid->begin();
	while(!frontGrid->end() ) {
		
		center = frontGrid->coordToCellCenter(frontGrid->key() );
		
		for(int i=0; i<6; ++i) {
			cand.pos = center + Vector3F( frontSize * PoissonSequence<Disk>::TwentySixNeighborCoord[i][0],
										 frontSize * PoissonSequence<Disk>::TwentySixNeighborCoord[i][1],
										 frontSize * PoissonSequence<Disk>::TwentySixNeighborCoord[i][2]);
			fillBackgroudAt(dst, frontGrid, 
							cand, n);
		}

		frontGrid->next();
	}
}

bool SuperformulaPoisson::fillBackgroudAt(PoissonSequence<Disk> * dst,
						PoissonSequence<Disk> * frontGrid,
						Disk & cand,
						int & n)
{	
	bool rjt = false;
	sdb::Array<int, Disk > * frontCell = frontGrid->findCell(cand.pos);
/// only check no-empty front cells
	if(frontCell) {
		rjt = frontGrid->rejectIn(frontCell, &cand);
	}
	
	if(!rjt ) {
		rjt = dst->reject(&cand);
	}
		
	if(!rjt ) {
		
		Disk * samp = new Disk;
		samp->pos = cand.pos;
		samp->r = cand.r;
		samp->key = n;
		
		dst->insert((const float *)&cand.pos, samp);
		
		n++;
		return true;
	}
	return false;
}

void SuperformulaPoisson::draw(GeoDrawer * dr)
{
	dr->setColor(.125f, .3f, .15f);
	m_bkg.begin();
	while(!m_bkg.end() ) {
	
		dr->boundingBox(m_bkg.coordToGridBBox(m_bkg.key() ) );
		m_bkg.next();
	}
	
	SuperformulaTest::draw(dr);
}

void SuperformulaPoisson::drawSamplesIn(GeoDrawer * dr,
						sdb::Array<int, Disk > * cell)
{
	cell->begin();
	while(!cell->end() ) {
		
		Disk * v = cell->value();
		dr->cube(v->pos, .0125f);
		
		cell->next();
	}
}

PoissonSequence<Disk> * SuperformulaPoisson::sampleGrid()
{ return &m_bkg; }

void SuperformulaPoisson::extractSamplePos(aphid::Vector3F * dst)
{ extractPos(dst, &m_bkg); }

void SuperformulaPoisson::extractPos(aphid::Vector3F * dst, PoissonSequence<Disk> * grid)
{
	int c = 0;
	grid->begin();
	while(!grid->end() ) {
		
		extractPosIn(dst, c, grid->value() );
		grid->next();
	}
}

void SuperformulaPoisson::extractPosIn(aphid::Vector3F * dst, int & count,
					sdb::Array<int, Disk > * cell)
{
	cell->begin();
	while(!cell->end() ) {
		
		Disk * v = cell->value();
		dst[count++] = v->pos;
		
		cell->next();
	}
}

void SuperformulaPoisson::extractSamplePosIn(std::vector<Vector3F> & dst,
					sdb::Array<int, Disk > * cell)
{
	cell->begin();
	while(!cell->end() ) {
		
		Disk * v = cell->value();
		
		dst.push_back(v->pos);
		
		cell->next();
	}
}

}
