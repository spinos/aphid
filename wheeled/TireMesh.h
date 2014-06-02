/*
 *  TireMesh.h
 *  wheeled
 *
 *  Created by jian zhang on 6/1/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "MeshShape.h"
namespace caterpillar {
class TireMesh {
public:
	TireMesh();
	virtual ~TireMesh();
	
	btCollisionShape* create(const float & radiusMajor, const float & radiusMinor, const float & width);
protected:
	
private:

};
}