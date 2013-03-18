/*
 *  SelectionArray.h
 *  lapl
 *
 *  Created by jian zhang on 3/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <vector>
class Primitive;
class Geometry;
class Vertex;
class SelectionArray {
public:
	SelectionArray();
	virtual ~SelectionArray();
	
	void reset();
	void add(Primitive * prim);
	
	unsigned numPrims() const;
	Primitive * getPrimitive(const unsigned & idx) const;
	
	unsigned numVertices() const;
	Vertex * getVertex(const unsigned & idx) const;
private:
	bool isVertexSelected(unsigned idx) const;
	std::vector<Primitive *> m_prims;
	std::vector<Vertex *> m_vertices;
};