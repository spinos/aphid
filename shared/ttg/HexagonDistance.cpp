/*
 *  HexagonDistance.cpp
 *  
 *
 *  Created by zhang on 17-2-4.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "HexagonDistance.h"
#include <geom/ConvexShape.h>

namespace aphid {

namespace ttg {

HexagonDistance::HexagonDistance()
{}

HexagonDistance::~HexagonDistance()
{}

const float * HexagonDistance::result() const
{ return m_d; }

const bool * HexagonDistance::isValid() const
{ return m_valid; }

}

}

