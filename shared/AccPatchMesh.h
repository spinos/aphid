/*
 *  AccPatchMesh.h
 *  mallard
 *
 *  Created by jian zhang on 8/30/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <PatchMesh.h>
class AccPatch;
class MeshTopology;
class AccPatchMesh : public PatchMesh {
public:
	AccPatchMesh();
	virtual ~AccPatchMesh();
	
	void setup(MeshTopology * topo);
	
	AccPatch* beziers() const;
	
	virtual const BoundingBox calculateBBox(const unsigned &idx) const;
	virtual char intersect(unsigned idx, const Ray & ray, IntersectionContext * ctx) const;
	
private:
	AccPatch* m_bezier;
};
