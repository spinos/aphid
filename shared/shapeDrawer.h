/*
 *  shapeDrawer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

class ShapeDrawer {
public:
	ShapeDrawer () {}
	virtual ~ShapeDrawer () {}
	
	void box(float width, float height, float depth);
	void solidCube(float x, float y, float z, float size);
	void setGrey(float g);
};