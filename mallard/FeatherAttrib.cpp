/*
 *  FeatherAttrib.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherAttrib.h"

FeatherAttrib::FeatherAttrib() 
{
	m_resShaft = 10;
	m_resBarb = 9;
	m_numSeparate = 2;
	m_seed = 1;
	m_fuzzy = 0.f; 
	m_separateStrength = 0.f;
	m_barbShrink = .5f;
	m_shaftShrink = .5f;
	m_barbWidthScale = .67f;
}
