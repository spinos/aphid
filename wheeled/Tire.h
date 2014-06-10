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
    #define NUMGRIDRAD 30
    #define NUMPAD 60
	Tire();
	virtual ~Tire();
	
	btRigidBody* create(const float & radiusMajor, const float & radiusMinor, const float & width, const float & mass, const Matrix44F & tm, bool isLeft);
	void setFriction(const float & x);
protected:
	void attachPad(btRigidBody* wheelBody, btCollisionShape * padShape, const Matrix44F & origin, const float & radiusMajor, const float & radiusMinor, bool isLeft);
private:
    float m_hw;
    btRigidBody * m_bd[NUMPAD];
};
}