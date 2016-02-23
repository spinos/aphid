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
namespace aphid {

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

void AStripedModel::copyPointDrift(unsigned * src, unsigned n, unsigned start, unsigned offset)
{ 
    unsigned * dst = &pointDrifts()[start];
    unsigned i = 0;
    for(;i<n;i++) dst[i] = src[i] + offset;
}
 
void AStripedModel::copyIndexDrift(unsigned * src, unsigned n, unsigned start, unsigned offset)
{ 
    unsigned * dst = &indexDrifts()[start];
    unsigned i = 0;
    for(;i<n;i++) dst[i] = src[i] + offset;
}

}
//:~