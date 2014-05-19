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
	m_chassis.setOrigin(Vector3F(0.f, 6.f, -10.f));
	m_chassis.setSpan(84.f);
	m_chassis.setHeight(2.f);
	m_chassis.setWidth(24.f);
	m_chassis.setNumRoadWheels(7);
	m_chassis.setRoadWheelZ(0, 29.f);
	m_chassis.setRoadWheelZ(1, 18.f);
	m_chassis.setRoadWheelZ(2, 8.f);
	m_chassis.setRoadWheelZ(3, -2.f);
	m_chassis.setRoadWheelZ(4, -12.f);
	m_chassis.setRoadWheelZ(5, -22.f);
	m_chassis.setRoadWheelZ(6, -32.f);
	m_chassis.setNumSupportRollers(3);
	m_chassis.setSupportRollerZ(0, 24.f);
	m_chassis.setSupportRollerZ(1, 2.f);
	m_chassis.setSupportRollerZ(2, -18.f);
	createChassis(m_chassis);
	
	m_leftTread.setOrigin(m_chassis.trackOrigin());
	m_leftTread.setRadius(5.f);
	m_leftTread.setWidth(m_chassis.trackWidth());
	m_leftTread.setSpan(m_chassis.span());
	m_leftTread.computeNumShoes();
	createTread(m_leftTread);

	m_rightTread.setOrigin(m_chassis.trackOrigin(false));
	m_rightTread.setRadius(5.f);
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
		Matrix33F rot = tread.currentSpace().rotation();
		trans = btTransform(btMatrix3x3(rot.M(0, 0), rot.M(1, 0), rot.M(2, 0), rot.M(0, 1), rot.M(1, 1), rot.M(2, 1), rot.M(0, 2), rot.M(1, 2), rot.M(2, 2)));
		trans.setOrigin(btVector3(at.x, at.y, at.z));
		if(tread.currentIsShoe()) 
			curBody = createRigitBody(shoeShape, trans, 6.5f);
		else
			curBody = createRigitBody(pinShape, trans, 5.2f);
			
		curBody->setDamping(0.0f, 1.0f);
		curBody->setFriction(1.0f);
		
		if(!firstBody) firstBody = curBody;
			
		if(preBody) {
			btTransform frameInA(zToX), frameInB(zToX);
	
			if(tread.currentIsShoe()) {
				frameInA.setOrigin(btVector3(0.0, 0.0, pinZ * tread.PinHingeFactor));
				frameInB.setOrigin(btVector3(0.0, shoeY * tread.ShoeHingeRise, shoeZ *  -tread.ShoeHingeFactor));
				
				constrainByHinge(*preBody, *curBody, frameInA, frameInB, true);
				
				frameInA.setOrigin(btVector3(-tread.width() * 0.5f, 0.0, pinZ * tread.PinHingeFactor));
				frameInB.setOrigin(btVector3(-tread.width() * 0.5f, shoeY * tread.ShoeHingeRise, shoeZ *  -tread.ShoeHingeFactor));
				
				constrainByHinge(*preBody, *curBody, frameInA, frameInB, true);
				
				frameInA.setOrigin(btVector3(tread.width() * 0.5f, 0.0, pinZ * tread.PinHingeFactor));
				frameInB.setOrigin(btVector3(tread.width() * 0.5f, shoeY * tread.ShoeHingeRise, shoeZ *  -tread.ShoeHingeFactor));
				
				constrainByHinge(*preBody, *curBody, frameInA, frameInB, true);
			}
			else {
				frameInA.setOrigin(btVector3(0.0, shoeY * tread.ShoeHingeRise, shoeZ *  tread.ShoeHingeFactor));
				frameInB.setOrigin(btVector3(0.0, 0.0, pinZ *  -tread.PinHingeFactor));
				
				constrainByHinge(*preBody, *curBody, frameInA, frameInB, true);
				
				frameInA.setOrigin(btVector3(-tread.width() * 0.5f, shoeY * tread.ShoeHingeRise, shoeZ *  tread.ShoeHingeFactor));
				frameInB.setOrigin(btVector3(-tread.width() * 0.5f, 0.0, pinZ *  -tread.PinHingeFactor));
				
				constrainByHinge(*preBody, *curBody, frameInA, frameInB, true);
				
				frameInA.setOrigin(btVector3(tread.width() * 0.5f, shoeY * tread.ShoeHingeRise, shoeZ *  tread.ShoeHingeFactor));
				frameInB.setOrigin(btVector3(tread.width() * 0.5f, 0.0, pinZ *  -tread.PinHingeFactor));
				
				constrainByHinge(*preBody, *curBody, frameInA, frameInB, true);

			}
		}
		
		preBody = curBody;
		tread.next();
	}
	
	btTransform frameInShoe(zToX), frameInPin(zToX);
	frameInShoe.setOrigin(btVector3(0.f, shoeY * tread.ShoeHingeRise, shoeZ *  -tread.ShoeHingeFactor));
	frameInPin.setOrigin(btVector3(0.f, 0.0, pinZ * tread.PinHingeFactor));
	
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
	btRigidBody* chassisBody = createRigitBody(chassisShape, trans, 1000.f);
	chassisBody->setDamping(0.f, 0.f);
	createDriveSprocket(c, chassisBody);
	createDriveSprocket(c, chassisBody, false);
	createTensioner(c, chassisBody);
	createTensioner(c, chassisBody, false);
	createRoadWheels(c, chassisBody);
	createRoadWheels(c, chassisBody, false);
	createSupportRollers(c, chassisBody);
	createSupportRollers(c, chassisBody, false);
}

void TrackedPhysics::createWheel(CreateWheelProfile & profile)
{
	btCollisionShape* wheelShape = createCylinderShape(profile.radius, profile.width * .5f, profile.radius);
	const btMatrix3x3 yTonX(0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	const btMatrix3x3 yToX(0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	
	btTransform trans(yTonX);
	if(!profile.isLeft) trans = btTransform(yToX);
	trans.setOrigin(btVector3(profile.worldP.x, profile.worldP.y, profile.worldP.z));
	btRigidBody* wheelBody = createRigitBody(wheelShape, trans, profile.mass);
	wheelBody->setDamping(.0f, .0f);
	wheelBody->setFriction(1.0f);
	
	const btMatrix3x3 zToX(0.f, 0.f, -1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f);
	const btMatrix3x3 zTonX(0.f, 0.f, 1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 0.f);
	
	btTransform frameInA(zTonX);
	if(!profile.isLeft) frameInA = btTransform(zToX);
	
	frameInA.setOrigin(btVector3(profile.objectP.x, profile.objectP.y, profile.objectP.z));
	
	const btMatrix3x3 zTonY(1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, -1.f, 0.f);
	
	btTransform frameInB(zTonY);
	btGeneric6DofConstraint* hinge = constrainByHinge(*profile.connectTo, *wheelBody, frameInA, frameInB, true);
	profile.dstBody = wheelBody;
	profile.dstHinge = hinge;
}

void TrackedPhysics::createDriveSprocket(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.driveSprocketRadius();
	cwp.width = c.trackWidth();
	cwp.mass = 24.f;
	cwp.worldP = c.driveSprocketOrigin(isLeft);
	cwp.objectP = c.driveSprocketOriginObject(isLeft);
	cwp.isLeft = isLeft;
	
	createWheel(cwp);
	btGeneric6DofConstraint* hinge = cwp.dstHinge;
	//if(isLeft) hinge->enableAngularMotor(true, -10.f, 100.f);
	//else hinge->enableAngularMotor(true, 10.f, 100.f);
	if(isLeft) m_drive[0] = hinge;
	else m_drive[1] = hinge;
	//}
}

void TrackedPhysics::createTensioner(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.tensionerRadius();
	cwp.width = c.trackWidth();
	cwp.mass = 4.f;
	cwp.worldP = c.tensionerOrigin(isLeft);
	cwp.objectP = c.tensionerOriginObject(isLeft);
	cwp.isLeft = isLeft;
	
	createWheel(cwp);
	if(isLeft) m_tension[0] = cwp.dstHinge;
	else m_tension[1] = cwp.dstHinge;
	
}

void TrackedPhysics::createRoadWheels(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	if(c.numRoadWheels() < 1) return;
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.roadWheelRadius();
	cwp.width = c.trackWidth() * .2f;
	cwp.mass = 4.f;
	cwp.isLeft = isLeft;
	for(int i=0; i < c.numRoadWheels(); i++) {
		cwp.worldP = c.roadWheelOrigin(i, isLeft);
		cwp.objectP = c.roadWheelOriginObject(i, isLeft);
		createWheel(cwp);
		m_bearing.push_back(cwp.dstHinge);
	}
}

void TrackedPhysics::createSupportRollers(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	if(c.numSupportRollers() < 1) return;
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.supportRollerRadius();
	cwp.width = c.trackWidth() * .25f;
	cwp.mass = 1.f;
	cwp.isLeft = isLeft;
	for(int i=0; i < c.numSupportRollers(); i++) {
		cwp.worldP = c.supportRollerOrigin(i, isLeft);
		cwp.objectP = c.supportRollerOriginObject(i, isLeft);
		createWheel(cwp);
	}
}

void TrackedPhysics::addTension(const float & x)
{
	btGeneric6DofConstraint* te = m_tension[0];
	btTransform & frm = te->getFrameOffsetA();
	btVector3 & p = frm.getOrigin();
	if(p[2] < 43)p[2] += x;
	
	te = m_tension[1];
	btTransform & frm1 = te->getFrameOffsetA();
	btVector3 & p1 = frm1.getOrigin();
	if(p[2] < 43)p1[2] += x;
	
	std::deque<btGeneric6DofConstraint*>::iterator it = m_bearing.begin();
	for(; it != m_bearing.end(); ++it) {
		btTransform & fa = (*it)->getFrameOffsetA();
		btVector3 & p = fa.getOrigin();
		if(p[1] > -6)p[1] -= x;
		
	}
	
	m_drive[0]->getRotationalLimitMotor(2)->m_enableMotor = true;
	m_drive[0]->getRotationalLimitMotor(2)->m_targetVelocity -= x;
	m_drive[0]->getRotationalLimitMotor(2)->m_maxMotorForce += x * 100.f;
	m_drive[0]->getRotationalLimitMotor(2)->m_damping = 0.5f;
	m_drive[1]->getRotationalLimitMotor(2)->m_enableMotor = true;
	m_drive[1]->getRotationalLimitMotor(2)->m_targetVelocity += x;
	m_drive[1]->getRotationalLimitMotor(2)->m_maxMotorForce += x * 100.1f;
	m_drive[1]->getRotationalLimitMotor(2)->m_damping = 0.5f;
}
