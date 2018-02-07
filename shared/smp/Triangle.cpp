/*
 *  Triangle.cpp
 *  
 *
 *  Created by jian zhang on 2/10/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "Triangle.h"

namespace aphid {

namespace smp {

Triangle::Triangle() :
m_area(1.f)
{}

void Triangle::setSampleSize(float x)
{ m_area = x * x * .73f; }

int Triangle::getNumSamples(const float& tarea)
{ return 1 + tarea / m_area; }

}

}