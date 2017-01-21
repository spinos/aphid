/*
 *  DrawCircle.h
 *
 *
 *  Created by jian zhang on 2/7/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_CIRCLE
#define APH_OGL_DRAW_CIRCLE

namespace aphid {

class DrawCircle {
	
public:
	DrawCircle();
	virtual ~DrawCircle();
	
	void drawZCircle(const float * mat) const;
	void drawZRing(const float * mat) const;
	void drawCircle(const float * mat) const;
	void draw3Circles(const float * mat) const;
	
protected:
	void drawXRing() const;
	void drawYRing() const;
	void drawZRing() const;
	
private:
	void drawCircle() const;
	
	
};

}
#endif
