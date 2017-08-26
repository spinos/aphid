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
m_exclR(1.f),
m_blkType(StemBlock::tUnknown),
m_hasTerminalStem(false)
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

void StemBlock::setIsAxial()
{ m_blkType = tAxialStem; }

void StemBlock::setIsLeaf()
{ m_blkType = tLeaf; }

void StemBlock::setHasTerminalStem()
{ m_hasTerminalStem = true; }

const int& StemBlock::age() const
{ return m_age; }

const int& StemBlock::geomInd() const
{ return m_geomInd; }

const float& StemBlock::exclR() const
{ return m_exclR; }

bool StemBlock::isAxial() const
{ return m_blkType == tAxialStem; }

bool StemBlock::isStem() const
{ return m_blkType <= tAxialStem; }

bool StemBlock::isLeaf() const
{ return m_blkType == tLeaf; }

const bool& StemBlock::hasTerminalStem() const
{ return m_hasTerminalStem; }

StemBlock* StemBlock::childStem(int i) const
{ return static_cast<StemBlock* >(child(i) ); }

int StemBlock::geomInstanceInd() const
{
	if(isStem() )
		return ((m_geomInd + 1)<<20 );
		
	return m_geomInd;
}
