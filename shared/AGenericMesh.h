/*
 *  AGenericMesh.h
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "Geometry.h"
class BaseBuffer;
class AGenericMesh : public Geometry {
public:
	AGenericMesh();
	virtual ~AGenericMesh();
	
	virtual const Type type() const;
	virtual const BoundingBox calculateBBox() const;
	
	const unsigned numPoints() const;
	const unsigned numIndices() const;
	Vector3F * points() const;
	unsigned * indices() const;
	unsigned * anchors() const;
	
protected:
	void createBuffer(unsigned np, unsigned ni);
	void setNumPoints(unsigned n);
	void setNumIndices(unsigned n);
	void resetAnchors(unsigned n);
private:
	BaseBuffer * m_points;
	BaseBuffer * m_indices;
	BaseBuffer * m_anchors;
	unsigned m_numPoints, m_numIndices;
};