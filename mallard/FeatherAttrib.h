/*
 *  FeatherAttrib.h
 *  mallard
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class FeatherAttrib {
public:
	FeatherAttrib();
	
	unsigned m_resShaft, m_resBarb, m_numSeparate, m_seed;
	float m_fuzzy, m_separateStrength;
};