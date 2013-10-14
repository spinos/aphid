/*
 *  bezierPatch.h
 *  catmullclark
 *
 *  Created by jian zhang on 10/26/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <AllMath.h>
class BoundingBox;

struct PatchSplitContext {
	Vector2F patchUV[4];
	void reset() {
		patchUV[0].set(0.f, 0.f);
		patchUV[1].set(1.f, 0.f);
		patchUV[2].set(1.f, 1.f);
		patchUV[3].set(0.f, 1.f);
	}
};

class BezierPatch
{
public:
	BezierPatch();
	virtual ~BezierPatch();
	virtual void setTexcoord(float* u, float* v, unsigned* idx);
	virtual void evaluateContolPoints();
	virtual void evaluateTangents();
	virtual void evaluateBinormals();
	void evaluateSurfacePosition(float u, float v, Vector3F * pos) const;
	void evaluateSurfaceTangent(float u, float v, Vector3F * tang) const;
	void evaluateSurfaceBinormal(float u, float v, Vector3F * binm) const;
	void evaluateSurfaceNormal(float u, float v, Vector3F * nor) const;
	void evaluateSurfaceTexcoord(float u, float v, Vector3F * tex) const;
	const BoundingBox controlBBox() const;
	void decasteljauSplit(BezierPatch *dst) const;
	void splitPatchUV(PatchSplitContext ctx, PatchSplitContext * child) const;
	void tangentFrame(float u, float v, Matrix33F & frm) const;
	
	Vector3F p(unsigned u, unsigned v) const;
	Vector3F normal(unsigned u, unsigned v) const;
	Vector2F tex(unsigned u, unsigned v) const;
	Vector3F _contorlPoints[16];
	Vector3F _normals[16];
	Vector3F _tangents[12];
	Vector3F _binormals[12];
	Vector2F _texcoords[4];
};