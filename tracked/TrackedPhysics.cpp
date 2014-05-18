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
	m_chassis.setOrigin(Vector3F(0.f, 16.f, -10.f));
	m_chassis.setSpan(75.f);
	m_chassis.setHeight(7.f);
	m_chassis.setWidth(20.f);
	m_chassis.setDriveSprocketRadius(4.f);
	m_chassis.setTensionerRadius(3.f);
	m_chassis.setTrackWidth(8.f);
	createChassis(m_chassis);
	
	m_leftTread.setOrigin(m_chassis.trackOrigin());
	m_leftTread.setRadius(8.f);
	m_leftTread.setWidth(m_chassis.trackWidth());
	m_leftTread.setSpan(m_chassis.span());
	m_leftTread.computeNumShoes();
	createTread(m_leftTread);

	m_rightTread.setOrigin(m_chassis.trackOrigin(false));
	m_rightTread.setRadius(8.f);
	m_rightTread.setWidth(m_chassis.trackWidth());
	m_rightTread.setSpan(m_chassis.span());
	int nsh = m_rightTread.computeNumShoes();
	std::cout<<" num shoes "<<nsh;
	createTread(m_rightTread);
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
			curBody = createRigitBody(shoeShape, trans, 1.f);
		else
			curBody = createRigitBody(pinShape, trans, 0.1f);
			
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

void TrackedPhysics::createChassis(Chassis & c)
{
	const Vector3F dims = c.extends() * .5f;
	btCollisionShape* chassisShape = createBoxShape(dims.x, dims.y, dims.z);
	
	const Vector3F origin = c.center();
	btTransform trans;
	trans.setIdentity();
	trans.setOrigin(btVector3(origin.x, origin.y, origin.z));
	btRigidBody* chassisBody = createRigitBody(chassisShape, trans, 100.f);
	createDriveSprocket(c, chassisBody);
	createDriveSprocket(c, chassisBody, false);
	createTensioner(c, chassisBody);
	createTensioner(c, chassisBody, false);
}

void TrackedPhysics::createDriveSprocket(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	btCollisionShape* sprocketShape = createCylinderShape(c.driveSprocketRadius(), c.trackWidth() * 0.5f, c.driveSprocketRadius());
	Vector3F origin = c.driveSprocketOrigin(isLeft);
	const btMatrix3x3 yTonX(0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	const btMatrix3x3 yToX(0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	
	btTransform trans(yTonX);
	if(!isLeft) trans = btTransform(yToX);
	trans.setOrigin(btVector3(origin.x, origin.y, origin.z));
	
	btRigidBody* sprocketBody = createRigitBody(sprocketShape, trans, 10.f);
	sprocketBody->setDamping(.6f, .6f);
	sprocketBody->setFriction(0.99f);
	
	const btMatrix3x3 zToX(0.f, 0.f, -1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f);
	const btMatrix3x3 zTonX(0.f, 0.f, 1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 0.f);
	
	
	btTransform frameInA(zTonX);
	if(!isLeft) frameInA = btTransform(zToX);
	origin = c.driveSprocketOriginObject(isLeft);
	frameInA.setOrigin(btVector3(origin.x, origin.y, origin.z));
	
	const btMatrix3x3 zToY(1.f, 0.f, 0.f, 0.f, 0.f, -1.f, 0.f, 1.f, 0.f);
	const btMatrix3x3 zTonY(1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, -1.f, 0.f);
	
	btTransform frameInB(zTonY);
	btHingeConstraint* hinge = constrainByHinge(*chassisBody, *sprocketBody, frameInA, frameInB, true);
	if(isLeft) hinge->enableAngularMotor(true, -1.f, 10.f);
	else hinge->enableAngularMotor(true, 1.f, 10.f);
}

void TrackedPhysics::createTensioner(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	btCollisionShape* tensionerShape = createCylinderShape(c.tensionerRadius(), c.trackWidth() * 0.5f, c.tensionerRadius());
	Vector3F origin = c.tensionerOrigin(isLeft);
	const btMatrix3x3 yTonX(0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	const btMatrix3x3 yToX(0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	
	btTransform trans(yTonX);
	if(!isLeft) trans = btTransform(yToX);
	trans.setOrigin(btVector3(origin.x, origin.y, origin.z));
	
	btRigidBody* tensionerBody = createRigitBody(tensionerShape, trans, 10.f);
	tensionerBody->setDamping(.6f, .6f);
	tensionerBody->setFriction(0.99f);
	
	const btMatrix3x3 zToX(0.f, 0.f, -1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f);
	const btMatrix3x3 zTonX(0.f, 0.f, 1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 0.f);
	
	btTransform frameInA(zTonX);
	if(!isLeft) frameInA = btTransform(zToX);
	origin = c.tensionerOriginObject(isLeft);
	frameInA.setOrigin(btVector3(origin.x, origin.y, origin.z));
	
	const btMatrix3x3 zToY(1.f, 0.f, 0.f, 0.f, 0.f, -1.f, 0.f, 1.f, 0.f);
	const btMatrix3x3 zTonY(1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, -1.f, 0.f);
	
	btTransform frameInB(zTonY);
	constrainByHinge(*chassisBody, *tensionerBody, frameInA, frameInB, true);
}
