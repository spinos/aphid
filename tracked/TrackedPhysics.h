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
#include "Tread.h"

class TrackedPhysics : public DynamicsSolver
{
public:
	TrackedPhysics();
	virtual ~TrackedPhysics();
	
protected:
	virtual void clientBuildPhysics();
private:
	void createTread(Tread & t);
private:
	Tread m_leftTread, m_rightTread;
};

