/*
 *  WheeledVehicle.cpp
 *  wheeled
 *
 *  Created by jian zhang on 5/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "WheeledVehicle.h"
namespace caterpillar {
WheeledVehicle::WheeledVehicle() {}
WheeledVehicle::~WheeledVehicle() {}
void WheeledVehicle::setOrigin(const Vector3F & p) { m_origin = p; }
void WheeledVehicle::create() {}
}