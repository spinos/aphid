/*
 *  DistantLight.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "DistantLight.h"

DistantLight::DistantLight() {}
DistantLight::~DistantLight() {}

const TypedEntity::Type DistantLight::type() const
{ return TDistantLight; }