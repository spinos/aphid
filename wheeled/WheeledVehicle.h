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
#include <AllMath.h>
namespace caterpillar {
class WheeledVehicle : public GroupId {
public:
	WheeledVehicle();
	virtual ~WheeledVehicle();
	void setOrigin(const Vector3F & p);
	void create();
private:
	Vector3F m_origin;
};
}