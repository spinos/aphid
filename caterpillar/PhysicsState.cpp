/*
 *  PhysicsState.cpp
 *  caterpillar
 *
 *  Created by jian zhang on 5/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "PhysicsState.h"
#include <DynamicsSolver.h>
namespace caterpillar {
DynamicsSolver * PhysicsState::engine = NULL;
PhysicsState::EngineStatus PhysicsState::engineStatus = sUnkown;
PhysicsState::PhysicsState() 
{
	engine = new DynamicsSolver;
}

PhysicsState::~PhysicsState() 
{
	delete engine;
}
}
