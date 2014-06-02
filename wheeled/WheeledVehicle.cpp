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
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(chassisShape, trans, 4.f);
	chassisBody->setDamping(0.f, 0.f);
	
	group("chassis").push_back(id);
	
	Suspension::ChassisBody = chassisBody;
	Suspension::ChassisOrigin = origin();
	
	for(int i = 0; i < numAxis(); i++) {
		btRigidBody* hubL = suspension(i).create(wheelOrigin(i));
		btRigidBody* hubR = suspension(i).create(wheelOrigin(i, false), false);
		
		wheel(i).createShape();
		btRigidBody* wheL = wheel(i).create(wheelTM(i));
		btRigidBody* wheR = wheel(i).create(wheelTM(i, false));
		btTransform frmA; frmA.setIdentity();
		frmA.getOrigin()[0] = suspension(i).wheelHubX();
		
		btTransform frmB; frmB.setIdentity();
		btGeneric6DofConstraint* drv = PhysicsState::engine->constrainBy6Dof(*hubL, *wheL, frmA, frmB, true);
		drv->setAngularLowerLimit(btVector3(-SIMD_PI, 0.0, 0.0));
		drv->setAngularUpperLimit(btVector3(SIMD_PI, 0.0, 0.0));
		drv->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
		drv->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
		drv = PhysicsState::engine->constrainBy6Dof(*hubR, *wheR, frmA, frmB, true);
		drv->setAngularLowerLimit(btVector3(-SIMD_PI, 0.0, 0.0));
		drv->setAngularUpperLimit(btVector3(SIMD_PI, 0.0, 0.0));
		drv->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
		drv->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	}
}

}