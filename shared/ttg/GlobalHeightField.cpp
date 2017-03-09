/*
 *  GlobalHeightField.cpp
 *  
 *
 *  Created by jian zhang on 3/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GlobalHeightField.h"

namespace aphid {

namespace ttg {

GlobalHeightField::GlobalHeightField()
{
	m_planetCenter.set(0.f, -3.3895e6f, 0.f);/// mean radius of Mars
}

void GlobalHeightField::setPlanetRadius(float x)
{ m_planetCenter.y = -x; }

float GlobalHeightField::sample(const Vector3F & pos) const
{
	return pos.distanceTo(m_planetCenter) + m_planetCenter.y;
}

}

}