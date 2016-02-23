/*
 *  DrawCircle.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/7/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <CircleCurve.h>

class DrawCircle {
	static aphid::CircleCurve UnitCircleCurve;
	
public:
	DrawCircle();
	virtual ~DrawCircle();
	
protected:
	void drawCircle(const float * mat) const;
	void draw3Circles(const float * mat) const;
	void drawCircle() const;
	
private:

};
