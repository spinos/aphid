/*
 *  bezierPatch.h
 *  catmullclark
 *
 *  Created by jian zhang on 10/26/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "LODQuad.h"

class BezierPatch : public LODQuad
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
	//int getLODBase() const;
	
	Vector3F p(unsigned u, unsigned v) const;
	Vector3F normal(unsigned u, unsigned v) const;
	Vector2F tex(unsigned u, unsigned v) const;
	Vector3F _contorlPoints[16];
	Vector3F _normals[16];
	Vector3F _tangents[12];
	Vector3F _binormals[12];
	Vector2F _texcoords[4];
	//int _lodBase;
};
