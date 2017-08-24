/*
 *  StemBlock.cpp
 *  
 *
 *  Created by jian zhang on 8/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "StemBlock.h"

using namespace aphid;

StemBlock::StemBlock(StemBlock* parent) : deform::Block(parent),
m_geomInd(0),
m_exclR(1.f)
{
	if(parent) {
		m_age = parent->age() + 1;
		
	} else
		m_age = 0;
}

void StemBlock::setGeomInd(int x)
{ m_geomInd = x; }

void StemBlock::setExclR(float x)
{ m_exclR = x; }

const int& StemBlock::age() const
{ return m_age; }

const int& StemBlock::geomInd() const
{ return m_geomInd; }

const float& StemBlock::exclR() const
{ return m_exclR; }

StemBlock* StemBlock::childStem(int i) const
{ return static_cast<StemBlock* >(child(i) ); }
