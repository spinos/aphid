/*
 *  shapeDrawer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <QGLWidget>
#include "shapeDrawer.h"

inline void glDrawVector(const btVector3& v) { glVertex3d(v[0], v[1], v[2]); }


void ShapeDrawer::draw(btScalar* m, const btCollisionShape* shape)
{
	btVector3 org(m[12], m[13], m[14]);
	btVector3 dx(m[0], m[1], m[2]);
	btVector3 dy(m[4], m[5], m[6]);
	btVector3 dz(m[8], m[9], m[10]);
	const btBoxShape* boxShape = static_cast<const btBoxShape*>(shape);
	btVector3 halfExtent = boxShape->getHalfExtentsWithMargin();
	dx *= halfExtent[0];
	dy *= halfExtent[1];
	dz *= halfExtent[2];
	glColor3f(0.f, 0.f, 1.f);
	glBegin(GL_LINE_LOOP);
	glDrawVector(org - dx - dy - dz);
	glDrawVector(org + dx - dy - dz);
	glDrawVector(org + dx + dy - dz);
	glDrawVector(org - dx + dy - dz);
	glDrawVector(org - dx + dy + dz);
	glDrawVector(org + dx + dy + dz);
	glDrawVector(org + dx - dy + dz);
	glDrawVector(org - dx - dy + dz);
	glEnd();
	glBegin(GL_LINES);
	glDrawVector(org + dx - dy - dz);
	glDrawVector(org + dx - dy + dz);
	glDrawVector(org + dx + dy - dz);
	glDrawVector(org + dx + dy + dz);
	glDrawVector(org - dx - dy - dz);
	glDrawVector(org - dx + dy - dz);
	glDrawVector(org - dx - dy + dz);
	glDrawVector(org - dx + dy + dz);
	glEnd();
}