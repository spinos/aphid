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
#include <Patch.h>
#include <AccPatchGroup.h>
class BezierPatch;
class AccPatch;
class MeshTopology;
struct PatchSplitContext;

class AccPatchMesh : public PatchMesh, public AccPatchGroup {
public:
	AccPatchMesh();
	virtual ~AccPatchMesh();
	
	void setup(MeshTopology * topo);
	void update(MeshTopology * topo);
	
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(const unsigned &idx) const;
	virtual char intersect(unsigned idx, IntersectionContext * ctx) const;
	virtual char closestPoint(unsigned idx, const Vector3F & origin, IntersectionContext * ctx);
	virtual void pushPlane(unsigned idx, Patch::PushPlaneContext * ctx) const;
	
	void pointOnPatch(unsigned idx, float u, float v, Vector3F & dst) const;
	void normalOnPatch(unsigned idx, float u, float v, Vector3F & dst) const;
	void texcoordOnPatch(unsigned idx, float u, float v, Vector3F & dst) const;
	void tangentFrame(unsigned idx, float u, float v, Matrix33F & frm) const;
	
	void getPatchHir(unsigned idx, std::vector<Vector3F> & dst);
private:
	char recursiveBezierIntersect(BezierPatch* patch, IntersectionContext * ctx, const PatchSplitContext split, int level) const;
	void recursiveBezierClosestPoint(const Vector3F & origin, BezierPatch* patch, IntersectionContext * ctx, const PatchSplitContext split, int level) const;
	void recursiveBezierPushPlane(BezierPatch* patch, Patch::PushPlaneContext * ctx, int level) const;
};
