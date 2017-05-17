/*
 *  GardenExamp.cpp
 *  rhizoid
 *
 *  Created by jian zhang on 5/12/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GardenExamp.h"
#include "CompoundExamp.h"

namespace aphid {

GardenExamp::GardenExamp()
{}

GardenExamp::~GardenExamp()
{}

void GardenExamp::addAExample(CompoundExamp * v)
{ m_examples.push_back(v); }

int GardenExamp::numExamples() const
{ return m_examples.size(); }

CompoundExamp * GardenExamp::getCompoundExample(const int & i)
{ return m_examples[i]; }

const ExampVox * GardenExamp::getExample(const int & i) const
{ return m_examples[i]; }

}