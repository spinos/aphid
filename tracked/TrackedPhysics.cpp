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
	m_leftTread.setOrigin(Vector3F(15.f, 16.f, -10.f));
	m_leftTread.setRadius(7.9f);
	m_leftTread.setWidth(8.9f);
	m_leftTread.setSpan(78.f);
	m_leftTread.computeNumShoes();
	createTread(m_leftTread);

	m_rightTread.setOrigin(Vector3F(-15.f, 16.f, -10.f));
	m_rightTread.setRadius(7.9f);
	m_rightTread.setWidth(8.9f);
	m_rightTread.setSpan(78.f);
	int nsh = m_rightTread.computeNumShoes();
	std::cout<<" num shoes "<<nsh;
	//createTread(m_rightTread);
}

void TrackedPhysics::createTread(Tread & tread)
{
	const float pinX = tread.width() * 0.5f;
	const float shoeX = pinX * tread.ShoeWidthFactor;
	const float shoeZ = tread.shoeLength() * 0.5f * tread.ShoeLengthFactor;
	const float pinZ = tread.shoeLength() * 0.5f * tread.PinToShoeLengthRatio;
	const float shoeY = shoeZ * tread.ShoeThickness;
	const float pinY = pinZ * tread.PinThickness;
	
	btCollisionShape* shoeShape = createBoxShape(shoeX, shoeY, shoeZ);
	btCollisionShape* pinShape = createBoxShape(pinX, pinY, pinZ);

	btRigidBody* preBody = NULL;
	btRigidBody* firstBody = NULL;
	btRigidBody* curBody = NULL;
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
			
		curBody->setDamping(.6f, .6f);
		curBody->setFriction(0.9f);
			
		if(!firstBody) firstBody = curBody;
			
		if(preBody) {
			btTransform frameInA(zToX), frameInB(zToX);
	
			if(tread.currentIsShoe()) {
				frameInA.setOrigin(btVector3(0.0, 0.0, pinZ * tread.PinHingeFactor));
				frameInB.setOrigin(btVector3(0.0, shoeY * tread.ShoeHingeRise, shoeZ *  -tread.ShoeHingeFactor));
			}
			else {
				frameInA.setOrigin(btVector3(0.0, shoeY * tread.ShoeHingeRise, shoeZ *  tread.ShoeHingeFactor));
				frameInB.setOrigin(btVector3(0.0, 0.0, pinZ *  -tread.PinHingeFactor));
			}
			
			constrainByHinge(*preBody, *curBody, frameInA, frameInB, true);
		}
		
		preBody = curBody;
		tread.next();
	}
	
	btTransform frameInShoe(zToX), frameInPin(zToX);
	frameInShoe.setOrigin(btVector3(0.0, shoeY * tread.ShoeHingeRise, shoeZ *  -tread.ShoeHingeFactor));
	frameInPin.setOrigin(btVector3(0.0, 0.0, pinZ * tread.PinHingeFactor));
	
	constrainByHinge(*curBody, *firstBody, frameInPin, frameInShoe, true);
}