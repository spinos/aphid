/*
 *  BezierDrawer.h
 *  knitfabric
 *
 *  Created by jian zhang on 6/4/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <BaseDrawer.h>
#include <BezierPatch.h>
#include <AccPatchMesh.h>

class Tessellator;

class BezierDrawer : public BaseDrawer {
public:
	BezierDrawer();
	virtual ~BezierDrawer();
	void updateMesh(AccPatchMesh * mesh);
	
	void drawBezierPatch(BezierPatch * patch);
	void drawBezierCage(BezierPatch * patch);
	void drawAcc() const;
	void verbose() const;
private:
	void cleanup();
private:
	Tessellator* m_tess;
	AccPatchMesh * m_mesh;
	Vector3F * m_vertices;
	Vector3F * m_normals;
	unsigned * m_indices;
};
