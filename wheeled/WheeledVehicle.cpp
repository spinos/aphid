/*
 *  WheeledVehicle.cpp
 *  wheeled
 *
 *  Created by jian zhang on 5/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "WheeledVehicle.h"
#include <DynamicsSolver.h>
#include <PhysicsState.h>
namespace caterpillar {
WheeledVehicle::WheeledVehicle() 
{
	addGroup("chassis");
}

WheeledVehicle::~WheeledVehicle() {}

void WheeledVehicle::create() 
{
	resetGroups();
	
	const Vector3F dims = getChassisDim() * .5f; 
	dims.verbose("hulldim");
	btCollisionShape* chassisShape = PhysicsState::engine->createBoxShape(dims.x, dims.y, dims.z);
	
	btTransform trans;
	trans.setIdentity();
	trans.setOrigin(btVector3(origin().x, origin().y, origin().z));
	
	const int id = PhysicsState::engine->numCollisionObjects();
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(chassisShape, trans, 10.f);
	chassisBody->setDamping(0.f, 0.f);
	
	group("chassis").push_back(id);
	
	for(int i = 0; i < numAxis(); i++) {
		suspension(i).create(wheelOrigin(i));
		suspension(i).create(wheelOrigin(i, false), false);
		
		wheel(i).createShape();
		wheel(i).create(wheelTM(i));
		wheel(i).create(wheelTM(i, false));
	}
}

}