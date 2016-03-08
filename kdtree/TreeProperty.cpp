/*
 *  TreeProperty.cpp
 *  testntree
 *
 *  Created by jian zhang on 3/8/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "TreeProperty.h"
#include <iostream>
#include <sstream>

namespace aphid {

TreeProperty::TreeProperty() :
m_maxLevel(0)
{}

TreeProperty::~TreeProperty() {}

void TreeProperty::resetPropery()
{
	m_maxLevel = 0;
	m_numEmptyNodes = 0;
	m_numInternalNodes = 0;
	m_numLeafNodes = 0;
	m_emptyVolume = 0.f;
	m_minPrims = 1<<28;
	m_maxPrims = 0;
	m_totalNPrim = 0;
}

void TreeProperty::addMaxLevel(int x)
{ if(x > m_maxLevel) m_maxLevel = x; }

void TreeProperty::addEmptyVolume(float x)
{ 
	m_numEmptyNodes++;
	m_emptyVolume += x;
}

void TreeProperty::setTotalVolume(float x)
{ m_totalVolume = x; }

void TreeProperty::addNInternal()
{ m_numInternalNodes++; }

void TreeProperty::addNLeaf()
{ m_numLeafNodes++; }

void TreeProperty::updateNPrim(int x)
{
	if(m_minPrims > x) m_minPrims = x;
	if(m_maxPrims < x) m_maxPrims = x;
	m_totalNPrim += x;
}

int TreeProperty::numNoEmptyLeaves() const
{ return m_numLeafNodes; }

std::string TreeProperty::logProperty() const
{
	std::stringstream sst;
	sst<<"\n max leaf(treelet) level "<<m_maxLevel
	<<"\n n inner node "<<m_numInternalNodes
	<<"\n n leaf/empty "<<m_numLeafNodes<<"/"<<m_numEmptyNodes
	<<"\n n prim min/max/average "<<m_minPrims<<"/"<<m_maxPrims
	<<"/"<<((float)m_totalNPrim/(float)m_numLeafNodes)
	<<"\n total volume "<<m_totalVolume
	<<"\n empty ratio  "<<m_emptyVolume/m_totalVolume;
	return sst.str();
}

}