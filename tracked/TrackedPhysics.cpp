/*
 *  TrackedPhysics.cpp
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "TrackedPhysics.h"

TrackedPhysics::TrackedPhysics() {}
TrackedPhysics::~TrackedPhysics() {}

void TrackedPhysics::clientBuildPhysics()
{
	btCollisionShape* cubeShape = createBoxShape(4.f, .2f, 1.f);
	
	btTransform trans;
	trans.setIdentity();
	
	trans.setOrigin(btVector3(3.5, 5.0, 2.0));
	
	createRigitBox(cubeShape, trans, 1.f);
	
	trans.setOrigin(btVector3(0.0, 10.0, 0.0));
	
	btRigidBody* a = createRigitBox(cubeShape, trans, 1.f);
	
	trans.setOrigin(btVector3(0.0, 10.0, 2.0));
	
	btRigidBody* b = createRigitBox(cubeShape, trans, 1.f);
	
	trans.setOrigin(btVector3(0.0, 10.0, 4.0));
	
	btRigidBody* c = createRigitBox(cubeShape, trans, 1.f);
	btMatrix3x3 zToX(0.f, 0.f, -1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f);
	
	btTransform frameInA(zToX), frameInB(zToX);
	
	frameInA.setOrigin(btVector3(0.0, 0.0, 1.5));
	frameInB.setOrigin(btVector3(0.0, 0.0, -1.1));
	
	constrainByHinge(*a, *b, frameInA, frameInB);

	frameInA.setOrigin(btVector3(0.0, 0.0, 1.1));
	frameInB.setOrigin(btVector3(0.0, 0.0, -1.5));
	
	constrainByHinge(*b, *c, frameInA, frameInB);
}