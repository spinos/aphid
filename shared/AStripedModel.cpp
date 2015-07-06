/*
 *  AStripedModel.cpp
 *  aphid
 *
 *  Created by jian zhang on 7/6/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AStripedModel.h"
#include <BaseBuffer.h>

AStripedModel::AStripedModel() 
{
	m_numStripes = 0;
	m_pDrift = new BaseBuffer;
	m_iDrift = new BaseBuffer;
}

AStripedModel::~AStripedModel()
{
	delete m_pDrift;
	delete m_iDrift;
}

void AStripedModel::create(unsigned n)
{
	m_pDrift->create(n*4);
	m_iDrift->create(n*4);
	m_numStripes = n;
}

unsigned * AStripedModel::pointDrifts()
{ return (unsigned *)m_pDrift->data(); }

unsigned * AStripedModel::indexDrifts()
{ return (unsigned *)m_iDrift->data(); }

const unsigned AStripedModel::numStripes() const
{ return m_numStripes; }
//:~