/*
 *  ForestGrid.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/18/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ForestGrid.h"
#include <PlantCommon.h>

namespace aphid {

ForestGrid::ForestGrid()
{ 
	m_numActiveCells = 0;
	m_numActiveSamples = 0; 
	m_numVisibleSamples = 0;
}

ForestGrid::~ForestGrid()
{}

const int & ForestGrid::numActiveCells() const
{ return m_numActiveCells; }

const int & ForestGrid::numActiveSamples() const
{ return m_numActiveSamples; }

const int & ForestGrid::numVisibleSamples() const
{ return m_numVisibleSamples; }

void ForestGrid::activeCellBegin()
{ m_activeCells.begin(); }

void ForestGrid::activeCellNext()
{ m_activeCells.next(); }

const bool ForestGrid::activeCellEnd() const
{ return m_activeCells.end(); }

ForestCell * ForestGrid::activeCellValue()
{ return m_activeCells.value(); }

const sdb::Coord3 & ForestGrid::activeCellKey() const
{ return m_activeCells.key(); }

void ForestGrid::deselectCells()
{
	if(m_numActiveCells < 1) {
		return;
	}
	
	m_activeCells.begin();
	while(!m_activeCells.end() ) {
		m_activeCells.value()->deselectSamples();
		m_activeCells.next();
	}
	m_activeCells.clearSequence();
	m_numActiveCells = 0;
}

int ForestGrid::countActiveSamples()
{
	m_numActiveSamples = 0;
	m_numVisibleSamples = 0;
	
	if(m_numActiveCells < 1) {
		return 0;
	}
	
	m_activeCells.begin();
	while(!m_activeCells.end() ) {
		const ForestCell * cell = m_activeCells.value();
		m_numActiveSamples += cell->numActiveSamples();
		m_numVisibleSamples += cell->numVisibleSamples();
		m_activeCells.next();
	}
	return m_numActiveSamples;
}

void ForestGrid::clearSamplles()
{
	deselectCells();
	begin();
	while(!end() ) {
		value()->clearSamples();
		next();
	}
}

int ForestGrid::countPlants()
{
	int c = 0;
	begin();
	while(!end() ) {
		c += value()->size();
		next();
	}
	return c;
}

void ForestGrid::reshuffleSamples(const int & level)
{
	std::cout<<"\n Forest reshuffle samples at level "<<level;

	m_activeCells.begin();
	while(!m_activeCells.end() ) {
		ForestCell * cell = m_activeCells.value();
		if(cell->hasSamples(level) ) {
			cell->reshuffleSamples(level);
		}
		
		m_activeCells.next();
	}
}

void ForestGrid::clearPlants()
{
	begin();
	while(!end() ) {
		value()->clearPlants();
		next();
	}
}

}