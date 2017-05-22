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

void CompoundExamp::addInstance(const float * tm, const int & instanceId)
{
	InstanceD ainst;
	memcpy(ainst._trans, tm, 64);
	ainst._exampleId = 0;
	ainst._instanceId = instanceId;
	m_instances.push_back(ainst);
}

int CompoundExamp::numInstances() const
{ return m_instances.size(); }

const ExampVox::InstanceD & CompoundExamp::getInstance(const int & i) const
{ return m_instances[i]; }

bool CompoundExamp::isCompound() const
{ return true; }

}