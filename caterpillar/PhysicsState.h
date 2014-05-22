/*
 *  PhysicsState.h
 *  caterpillar
 *
 *  Created by jian zhang on 5/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
class DynamicsSolver;
namespace caterpillar {
class PhysicsState {
public:
	enum EngineStatus {
		sUnkown = 0,
		sCreating = 1,
		sUpdating = 2
	};
	PhysicsState();
	virtual ~PhysicsState();
	
	static DynamicsSolver * engine;
	static EngineStatus engineStatus;
};
}