/*
 *  PotAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "PotAttribs.h"
#include <GrowthSample.h>
#include <gar_common.h>

PotAttribs::PotAttribs() : PieceAttrib(gar::gtPot)
{
	addFloatAttrib(gar::nGrowPortion, 1.f, 0.1f, 1.f);
	addFloatAttrib(gar::nGrowMargin, 1.f, 0.5f, 2.f);
	addFloatAttrib(gar::nZenithNoise, 0.2f, 0.f, 1.f);
}

bool PotAttribs::isGround() const
{ return true; }

void PotAttribs::getGrowthProfile(GrowthSampleProfile* prof) const
{
	findAttrib(gar::nGrowMargin)->getValue(prof->m_sizing);
	prof->m_sizing *= prof->_exclR;
	
	findAttrib(gar::nZenithNoise)->getValue(prof->m_zenithNoise);
	findAttrib(gar::nGrowPortion)->getValue(prof->m_portion);
	prof->m_portion *= .43f;
	prof->m_angle = -1.f;
	prof->m_numSampleLimit = 80;
	
}