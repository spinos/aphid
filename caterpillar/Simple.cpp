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

Simple::Simple() 
{
	addGroup("box_left");
	addGroup("box_right");
}

Simple::~Simple() {}

void Simple::setDim(const float & x, const float & y, const float & z)
{
	m_dim.set(x, y, z);
}

void Simple::create()
{
	resetGroups();
	int id = PhysicsState::engine->numCollisionObjects();
	btCollisionShape* obstacleShape = PhysicsState::engine->createBoxShape(m_dim.x, m_dim.y, m_dim.z);
	Matrix44F trans; trans.setTranslation(0.f,10.f,0.f);
	PhysicsState::engine->createRigidBody(obstacleShape, trans, 1.f);
	group("box_left").push_back(id);
	
	id = PhysicsState::engine->numCollisionObjects();
	trans.setTranslation(1.5f,13.f,0.f);
	PhysicsState::engine->createRigidBody(obstacleShape, trans, 1.f);
	group("box_right").push_back(id);
}
}