/*
 *  Tire.cpp
 *  wheeled
 *
 *  Created by jian zhang on 6/1/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Tire.h"
#include <DynamicsSolver.h>
#include "PhysicsState.h"
#include <Common.h>
namespace caterpillar {
#define NUMGRIDRAD 83
	
Tire::Tire() {}
Tire::~Tire() {}
btCollisionShape* Tire::create(const float & radiusMajor, const float & radiusMinor, const float & width)
{
	const float sy = (radiusMajor - radiusMinor) * PI * 2.f / NUMGRIDRAD;
	btCompoundShape* wheelShape = new btCompoundShape();
	btCollisionShape* rollShape = PhysicsState::engine->createBoxShape(width * .5f, sy, radiusMinor);
	const float delta = PI * 2.f / (float)NUMGRIDRAD;
	int i;
	Matrix44F rot;
	for(i = 0; i < NUMGRIDRAD; i++) {
		Matrix44F ctm;
		ctm.translate(0.f, 0.f, radiusMajor - radiusMinor);
		ctm *= rot;
		btTransform frm = Common::CopyFromMatrix44F(ctm);
		wheelShape->addChildShape(frm, rollShape);
	
		rot.rotateX(delta);
	}
	
	return wheelShape;
}
}