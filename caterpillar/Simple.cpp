/*
 *  Simple.cpp
 *  caterpillar
 *
 *  Created by jian zhang on 5/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Simple.h"
#include <DynamicsSolver.h>
#include "PhysicsState.h"
namespace caterpillar {

Simple::Simple() {}
Simple::~Simple() {}
void Simple::setDim(const float & x, const float & y, const float & z)
{
	m_dim.set(x, y, z);
}

void Simple::create()
{
	int id = PhysicsState::engine->numCollisionObjects();
	btCollisionShape* obstacleShape = PhysicsState::engine->createBoxShape(m_dim.x, m_dim.y, m_dim.z);
	btTransform trans; trans.setIdentity(); trans.setOrigin(btVector3(0,10,0));
	PhysicsState::engine->createRigitBody(obstacleShape, trans, 1.f);
	
	id = PhysicsState::engine->numCollisionObjects();
	trans.setOrigin(btVector3(0.5,13,0));
	PhysicsState::engine->createRigitBody(obstacleShape, trans, 1.f);
}
}