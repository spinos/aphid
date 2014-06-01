/*
 *  WheeledVehicle.h
 *  wheeled
 *
 *  Created by jian zhang on 5/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <GroupId.h>
#include "WheeledChassis.h"
namespace caterpillar {
class WheeledVehicle : public WheeledChassis, public GroupId {
public:
	WheeledVehicle();
	virtual ~WheeledVehicle();
	void create();
private:
	
};
}