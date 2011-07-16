/*
 *  shapeDrawer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "btBulletCollisionCommon.h"

class ShapeDrawer {
public:
	ShapeDrawer () {}
	virtual ~ShapeDrawer () {}
	
	virtual void draw(btScalar* m, const btCollisionShape* shape);
};