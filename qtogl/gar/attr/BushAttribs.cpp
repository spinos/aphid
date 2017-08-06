/*
 *  BushAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BushAttribs.h"

BushAttribs::BushAttribs()
{
	addFloatAttrib(gar::nGrowMargin, 1.f, 0.5f, 2.f);
	addFloatAttrib(gar::nGrowAngle, 0.5f, 0.1f, 1.f);
	addFloatAttrib(gar::nZenithNoise, 0.2f, 0.f, 1.f);
}