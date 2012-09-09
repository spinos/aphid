/*
 *  Polytode.h
 *  convexHull
 *
 *  Created by jian zhang on 9/10/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <Vertex.h>
#include <Facet.h>
#include <vector>

class Polytode {
public:
	Polytode();
	virtual ~Polytode();
	
	void destroy();
	
	int getNumVertex() const;
	int getNumFace() const;
	
	void addVertex(Vertex *p);
	Vertex getVertex(int idx) const;
	Vertex *vertex(int idx);
	
	void addFacet(Facet *f);
	Facet getFacet(int idx) const;
	void removeFaces();
	
	std::vector<Vertex *>m_vertices;
	std::vector<Facet *>m_faces;
};