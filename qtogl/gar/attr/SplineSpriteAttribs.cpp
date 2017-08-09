/*
 *  SplineSpriteAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SplineSpriteAttribs.h"
#include <iostream>

SplineSpriteAttribs::SplineSpriteAttribs()
{
std::cout<<"SplineSpriteAttribs";std::cout.flush();
	addFloatAttrib(gar::nWidth, 4.f, 2.f, 16.f);
	addFloatAttrib(gar::nHeight, 6.f, 3.f, 24.f);
	addSplineAttrib(gar::nLeftSide);
	addSplineAttrib(gar::nRightSide);
}
