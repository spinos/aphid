/*
 *  PointLight.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "PointLight.h"

PointLight::PointLight() {}
PointLight::~PointLight() {}

const TypedEntity::Type PointLight::type() const
{ return TPointLight; }