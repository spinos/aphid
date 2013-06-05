/*
 *  FabricDrawer.h
 *  knitfabric
 *
 *  Created by jian zhang on 6/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BezierDrawer.h>
class YarnPatch;
class FabricDrawer : public BezierDrawer {
public:
	FabricDrawer();
	
	void setPositions(Vector3F * p);
	void drawYarn(YarnPatch * patch);
	void drawWale(YarnPatch * patch);
private:
	Vector3F * m_positions;
};