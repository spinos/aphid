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
{ m_numActiveCells = 0; }

ForestGrid::~ForestGrid()
{}

const int & ForestGrid::numActiveCells() const
{ return m_numActiveCells; }

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


}