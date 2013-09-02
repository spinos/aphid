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

class PatchMesh : public BaseMesh {
public:
	PatchMesh();
	virtual ~PatchMesh();
	
	void prePatchUV(unsigned numUVs, unsigned numUVIds);
	
	unsigned numPatches() const;
	
	float * us();
	float * vs();
	unsigned * uvIds();
	
	virtual const BoundingBox calculateBBox(const unsigned &idx) const;
	virtual char intersect(unsigned idx, const Ray & ray, IntersectionContext * ctx) const;
	
	char planarIntersect(const Vector3F * fourCorners, const Ray & ray, IntersectionContext * ctx) const;
	virtual unsigned closestVertex(unsigned idx, const Vector3F & px) const;

private:
	unsigned m_numUVs, m_numUVIds;
	float * m_u;
	float * m_v;
	unsigned * m_uvIds;
};