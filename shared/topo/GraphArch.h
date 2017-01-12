/*
 *  GraphArch.h
 *  convexHull
 *
 *  Created by jian zhang on 9/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

namespace aphid {

class Vertex;
class Facet;

class GraphArch {
public:
	GraphArch();
	GraphArch(Facet *f, Vertex *v);
	virtual ~GraphArch();

	Vertex *vertex;
	Facet *face;
	
	GraphArch *previousVertex;
	GraphArch *previousFace;
	
	GraphArch *nextVertex;
	GraphArch *nextFace;
};

}
