/*
 *  shapeDrawer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "btBulletCollisionCommon.h"
#include "btBulletDynamicsCommon.h"
#include "BulletSoftBody/btSoftBody.h"
class ShapeDrawer {
public:
	ShapeDrawer () {}
	virtual ~ShapeDrawer () {}
	
	void drawConstraint(const btTypedConstraint* constraint);
	void drawObject(const btCollisionObject* object);
	void drawRigidBody(const btRigidBody* body);
	void drawSoftBody(const btSoftBody* body);
	void drawShape(const btCollisionShape* shape);
	void drawTransform(const btRigidBody & body);
	void loadWorldSpace(const btRigidBody* body);
	void drawTransform(const btTransform & t);
	void drawForce(const btRigidBody* body);
	void drawTranslateHandle(const btRigidBody* body);
	void drawAngularLimit(const btTransform& space, const btVector3& angularLower, const btVector3& angularUpper);
};