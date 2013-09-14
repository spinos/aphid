/*
 *  MlDrawer.h
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BezierDrawer.h>
#include "MlSkin.h"

class MlDrawer : public BezierDrawer {
public:
	MlDrawer();
	virtual ~MlDrawer();
	void drawFeather(MlSkin * skin);
private:
};