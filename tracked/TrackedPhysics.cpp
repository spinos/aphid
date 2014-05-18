/*
 *  TrackedPhysics.cpp
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "TrackedPhysics.h"

TrackedPhysics::TrackedPhysics() {}
TrackedPhysics::~TrackedPhysics() {}

void TrackedPhysics::clientBuildPhysics()
{
	m_leftTread.setOrigin(Vector3F(0.f, 10.f, -10.f));
	int nsh = m_leftTread.computeNumShoes();
	std::cout<<" num shoes "<<nsh;
	createTread(m_leftTread);

}

void TrackedPhysics::createTread(Tread & tread)
{
	const float pinX = tread.width() * 0.5f;
	const float shoeX = pinX * .9f;
	const float shoeZ = tread.shoeLength() * 0.5f * 0.87f;
	const float pinZ = shoeZ * 0.625f / 0.87f;
	const float shoeY = shoeZ * tread.ShoeThickness;
	const float pinY = pinZ * tread.PinThickness;
	
	btCollisionShape* shoeShape = createBoxShape(shoeX, shoeY, shoeZ);
	btCollisionShape* pinShape = createBoxShape(pinX, pinY, pinZ);

	
	btRigidBody* preBody = NULL;
	btRigidBody* curBody;
	btTransform trans;
	const btMatrix3x3 zToX(0.f, 0.f, -1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f);
	
	tread.begin();
	while(!tread.end()) {
		Vector3F at = tread.currentSpace().getTranslation();
		trans.setIdentity();
		trans.setOrigin(btVector3(at.x, at.y, at.z));
		if(tread.currentIsShoe()) 
			curBody = createRigitBox(shoeShape, trans, 1.f);
		else
			curBody = createRigitBox(pinShape, trans, 0.1f);
			
		if(preBody) {
			btTransform frameInA(zToX), frameInB(zToX);
	
			if(tread.currentIsShoe()) {
				frameInA.setOrigin(btVector3(0.0, 0.0, pinZ * 0.57));
				frameInB.setOrigin(btVector3(0.0, shoeY * 0.25, shoeZ *  -0.68));
				
			}
			else {
				frameInA.setOrigin(btVector3(0.0, shoeY * 0.25, shoeZ *  0.68));
				frameInB.setOrigin(btVector3(0.0, 0.0, pinZ *  -0.57));
			}
			
			constrainByHinge(*preBody, *curBody, frameInA, frameInB, true);
		}
		
		preBody = curBody;
		tread.next();
	}
}