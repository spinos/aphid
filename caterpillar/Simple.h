/*
 *  Simple.h
 *  caterpillar
 *
 *  Created by jian zhang on 5/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include "GroupId.h"
namespace caterpillar {
class Simple : public GroupId {
public:
	Simple();
	virtual ~Simple();
	void setDim(const float & x, const float & y, const float & z);
	void create();
private:
	Vector3F m_dim;
};

}