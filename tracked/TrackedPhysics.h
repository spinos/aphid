/*
 *  TrackedPhysics.h
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "DynamicsSolver.h"

class TrackedPhysics : public DynamicsSolver
{
public:
	TrackedPhysics();
	virtual ~TrackedPhysics();
	
protected:
	virtual void clientBuildPhysics();
};

