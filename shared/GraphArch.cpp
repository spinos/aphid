/*
 *  GraphArch.cpp
 *  convexHull
 *
 *  Created by jian zhang on 9/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "GraphArch.h"

GraphArch::GraphArch() {}
GraphArch::~GraphArch() {}

GraphArch::GraphArch(Facet *f, Vertex *v)
{
	vertex = v;
	face = f;
	previousVertex = 0;
	previousFace = 0;
	nextVertex = 0;
	nextFace = 0;
}