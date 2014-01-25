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
#include <DrawBuffer.h>
#include <BezierPatch.h>
#include <AccPatchMesh.h>

class Tessellator;

class BezierDrawer : public BaseDrawer, public DrawBuffer {
public:
	BezierDrawer();
	virtual ~BezierDrawer();
	
	virtual void rebuildBuffer(AccPatchMesh * mesh);
	void updateBuffer(AccPatchMesh * mesh);
	
	void drawBezierPatch(BezierPatch * patch);
	void drawBezierCage(BezierPatch * patch);

	void useTag(const std::string & name);
	
	void tagColor(AccPatchMesh * mesh);
private:
	std::string m_tagName;
	Tessellator* m_tess;
};
