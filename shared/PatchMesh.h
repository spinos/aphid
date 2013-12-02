/*
 *  PatchMesh.h
 *  catmullclark
 *
 *  Created by jian zhang on 5/13/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BaseMesh.h>
class PointInsidePolygonTest;
class PatchMesh : public BaseMesh {
public:
	PatchMesh();
	virtual ~PatchMesh();
	
	virtual unsigned getNumFaces() const;
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(const unsigned &idx) const;
	virtual char intersect(unsigned idx, IntersectionContext * ctx) const;
	virtual char closestPoint(unsigned idx, const Vector3F & origin, IntersectionContext * ctx) const;
	
	char patchIntersect(PointInsidePolygonTest & pa, IntersectionContext * ctx) const;
	virtual unsigned closestVertex(unsigned idx, const Vector3F & px) const;
	
	PointInsidePolygonTest patchAt(unsigned idx) const;
	void perVertexVectorOfPatch(unsigned idx, Vector3F * dst) const;
	void perVertexFloatOnPatch(unsigned idx, float u, float v, float * dst) const;
private:
	
};