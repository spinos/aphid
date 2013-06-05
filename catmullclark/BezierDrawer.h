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

class Tessellator;

class BezierDrawer : public BaseDrawer {
public:
	BezierDrawer();
	
	void drawBezierPatch(BezierPatch * patch);
	void drawBezierCage(BezierPatch * patch);
private:
	Tessellator* m_tess;
};
