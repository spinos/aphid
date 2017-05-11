/*
 *  CompoundExamp.cpp
 *  rhizoid
 *
 *  Created by jian zhang on 5/12/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "CompoundExamp.h"
#include <math/Matrix44F.h>

namespace aphid {

CompoundExamp::CompoundExamp()
{}

CompoundExamp::~CompoundExamp()
{}

void CompoundExamp::addInstance(const Matrix44F & tm, const int & instanceId)
{
	InstanceD ainst;
	tm.glMatrix(ainst._trans);
	ainst._exampleId = 0;
	ainst._instanceId = instanceId;
	m_instances.push_back(ainst);
}

int CompoundExamp::numInstances() const
{ return m_instances.size(); }

}