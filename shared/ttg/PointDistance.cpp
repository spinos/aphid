/*
 *  PointDistance.cpp
 *  
 *
 *  Created by zhang on 17-2-4.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "PointDistance.h"
#include <geom/ConvexShape.h>

namespace aphid {

namespace ttg {

PointDistance::PointDistance()
{}

PointDistance::~PointDistance()
{}

const float & PointDistance::result() const
{ return m_d; }

const bool & PointDistance::isValid() const
{ return m_valid; }
	
void PointDistance::setPos(const Vector3F & v)
{ m_pos = v; }
	
const Vector3F & PointDistance::PointDistance::pos() const
{ return m_pos; }

}

}

