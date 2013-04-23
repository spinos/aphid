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
#include <PrimitiveFilter.h>
class Vector3F;
class Primitive;
class Geometry;
class VertexPath;
class VertexAdjacency;

class SelectionArray : public PrimitiveFilter {
public:
	SelectionArray();
	virtual ~SelectionArray();
	
	void reset();
	void add(Geometry * geo, unsigned icomp);
	
	unsigned numPrims() const;
	Primitive * getPrimitive(const unsigned & idx) const;
	
	unsigned numVertices() const;
	unsigned getVertexId(const unsigned & idx) const;
	unsigned lastVertexId() const;
	Vector3F * getVertexP(const unsigned & idx) const;
	
	Geometry * getGeometry() const;
	
	unsigned numFaces() const;
	unsigned getFaceId(const unsigned & idx) const;
	
	void setTopology(VertexAdjacency * topo);
private:
	VertexPath * m_vertexPath;
    Geometry * m_geometry;
	bool isVertexSelected(unsigned idx) const;
	bool isFaceSelected(unsigned idx) const;
	std::vector<Primitive *> m_prims;
	std::vector<unsigned> m_vertexIds;
	std::vector<unsigned> m_faceIds;
};