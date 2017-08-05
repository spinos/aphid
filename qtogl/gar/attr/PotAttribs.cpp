/*
 *  PotAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "PotAttribs.h"

PotAttribs::PotAttribs()
{
	addFloatAttrib(gar::nGrowMargin, 0.8f, 0.5f, 1.5f);
	addFloatAttrib(gar::nZenithNoise, 0.5f);
}