/*
 *  DrawCircle.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/7/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "CircleCurve.h"
class DrawCircle {
	static CircleCurve UnitCircleCurve;
	
public:
	DrawCircle();
	virtual ~DrawCircle();
	
protected:
	void drawCircle(const float * mat) const;
	
private:

};