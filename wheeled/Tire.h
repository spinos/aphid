/*
 *  TireMesh.h
 *  wheeled
 *
 *  Created by jian zhang on 6/1/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Common.h>
namespace caterpillar {
class Tire {
public:
	Tire();
	virtual ~Tire();
	
	btCollisionShape* create(const float & radiusMajor, const float & radiusMinor, const float & width);
	void attachPad(btRigidBody* wheelBody, const Matrix44F & origin, const float & radiusMajor, const float & radiusMinor, bool isLeft);
protected:
	
private:
    btCollisionShape* m_padShape;
};
}