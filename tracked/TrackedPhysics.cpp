/*
 *  TrackedPhysics.cpp
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "TrackedPhysics.h"
#include <DynamicsSolver.h>
#include "PhysicsState.h"
namespace caterpillar {
#define CONTACTFRICTION 1.414
TrackedPhysics::TrackedPhysics() 
{ 
	addGroup("chassis");
	addGroup("left_track_shoe");
	addGroup("right_track_shoe");
	addGroup("left_track_pin");
	addGroup("right_track_pin");
	m_targeVelocity = 0.f;
}

TrackedPhysics::~TrackedPhysics() {}

void TrackedPhysics::create()
{
	resetGroups();
	// createObstacles();
	
	createChassis(*this);
	
	Tread::SprocketRadius = driveSprocketRadius();
	m_leftTread.setOrigin(trackOrigin());
	m_leftTread.setRadius(driveSprocketRadius() * 1.25f);
	m_leftTread.setWidth(trackWidth());
	m_leftTread.setSpan(span());
	m_leftTread.computeNumShoes();
	
	m_rightTread.setOrigin(trackOrigin(false));
	m_rightTread.setRadius(driveSprocketRadius() * 1.25f);
	m_rightTread.setWidth(trackWidth());
	m_rightTread.setSpan(span());
	m_rightTread.computeNumShoes();
	
	createTread(m_leftTread);
	createTread(m_rightTread);
	
	// setEnablePhysics(false);
	// setNumSubSteps(10);
}

void TrackedPhysics::createObstacles()
{
	btCollisionShape* obstacleShape = PhysicsState::engine->createBoxShape(40, 1, 4);
	btTransform trans; trans.setIdentity(); trans.setOrigin(btVector3(10,1,50));
	btRigidBody* obs = PhysicsState::engine->createRigitBody(obstacleShape, trans, 0.f);
	obs->setDamping(0,0);
	obs->setFriction(.5);
	trans.setOrigin(btVector3(-10,1,80));
	obs = PhysicsState::engine->createRigitBody(obstacleShape, trans, 0.f);
	obs->setDamping(0,0);
	obs->setFriction(.5);
}

void TrackedPhysics::createTread(Tread & tread)
{	
	const float shoeX = tread.shoeWidth() * 0.5f;
	const float shoeZ = tread.shoeLength() * 0.5f;
	const float shoeY = tread.shoeThickness() * 0.5f;
	
	btCollisionShape* shoeShape = createShoeShape(shoeX, shoeY, shoeZ);
	btCollisionShape* pinShape = createPinShape(tread);

	btRigidBody* preBody = NULL;
	btRigidBody* firstBody = NULL;
	btRigidBody* curBody = NULL;
	btTransform trans;
	const btMatrix3x3 zToX(0.f, 0.f, -1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f);
	
	const float hseg = tread.segLength() * .45f;
	
	tread.begin();
	while(!tread.end()) {
		Vector3F at = tread.currentSpace().getTranslation();
		Matrix33F rot = tread.currentSpace().rotation();
		trans = btTransform(btMatrix3x3(rot.M(0, 0), rot.M(1, 0), rot.M(2, 0), rot.M(0, 1), rot.M(1, 1), rot.M(2, 1), rot.M(0, 2), rot.M(1, 2), rot.M(2, 2)));
		trans.setOrigin(btVector3(at.x, at.y, at.z));
		if(tread.currentIsShoe()) {
			curBody = PhysicsState::engine->createRigitBody(shoeShape, trans, .8f);
			curBody->setFriction(CONTACTFRICTION);
		}
		else {
			curBody = PhysicsState::engine->createRigitBody(pinShape, trans, .7f);
			curBody->setFriction(0.);
		}
			
		curBody->setDamping(0.f, 1.0f);
		
		if(!firstBody) firstBody = curBody;
			
		if(preBody) {
			btTransform frameInA(zToX), frameInB(zToX);
	
			if(tread.currentIsShoe()) {
				frameInA.setOrigin(btVector3(0.0, 0.0, hseg * (1.f - Tread::PinHingeFactor)));
				frameInB.setOrigin(btVector3(0.0, shoeY * tread.ShoeHingeRise, hseg * -1.f * Tread::PinHingeFactor));
			}
			else {
				frameInA.setOrigin(btVector3(0.0, shoeY * tread.ShoeHingeRise, hseg * Tread::PinHingeFactor));
				frameInB.setOrigin(btVector3(0.0, 0.0, hseg * -1.f * (1.f - Tread::PinHingeFactor)));
			}
			threePointHinge(frameInA, frameInB, tread.width() * 0.5f, preBody, curBody);
		}
		
		preBody = curBody;
		tread.next();
	}
	
	btTransform frameInShoe(zToX), frameInPin(zToX);
	frameInShoe.setOrigin(btVector3(0.f, shoeY * tread.ShoeHingeRise, hseg * -1.f * Tread::PinHingeFactor));
	frameInPin.setOrigin(btVector3(0.f, 0.0, hseg * (1.f - Tread::PinHingeFactor)));


	threePointHinge(frameInPin, frameInShoe, tread.width() * 0.5f, curBody, firstBody);
}

void TrackedPhysics::threePointHinge(btTransform & frameInA, btTransform & frameInB, const float & side, btRigidBody* bodyA, btRigidBody* bodyB)
{
	PhysicsState::engine->constrainByHinge(*bodyA, *bodyB, frameInA, frameInB, true);
	
	btVector3 & p = frameInA.getOrigin();
	p[0] = -side;
	btVector3 & p1 = frameInB.getOrigin();
	p1[0] = -side;
	
	PhysicsState::engine->constrainByHinge(*bodyA, *bodyB, frameInA, frameInB, true);
	
	btVector3 & p2 = frameInA.getOrigin();
	p2[0] = side;
	btVector3 & p3 = frameInB.getOrigin();
	p3[0] = side;
	
	PhysicsState::engine->constrainByHinge(*bodyA, *bodyB, frameInA, frameInB, true);
}

btCollisionShape* TrackedPhysics::createShoeShape(const float & x, const float &y, const float & z)
{
	btCollisionShape* pad = PhysicsState::engine->createBoxShape(x, y, z);
	// btCollisionShape* tooth = createSphereShape(Tread::ToothWidth* .5f);
	btCollisionShape* tooth = PhysicsState::engine->createCylinderShape(Tread::ToothWidth* .5f, Tread::ToothHeight * .5f, Tread::ToothWidth* .5f);
	btCompoundShape* shoeShape = new btCompoundShape();
	
	btTransform childT; childT.setIdentity();
	shoeShape->addChildShape(childT, pad);
	
	childT.setOrigin(btVector3(0, Tread::ToothHeight * .5f,0));
	
	shoeShape->addChildShape(childT, tooth);
	return shoeShape;
}

btCollisionShape* TrackedPhysics::createPinShape(Tread & tread)
{
	const float pinX = tread.width() * 0.5f;
	const float pinZ = tread.pinLength() * 0.5f;
	const float pinY = tread.pinThickness() * 0.5f;
	
	btCollisionShape* pad = PhysicsState::engine->createBoxShape(pinX, pinY, pinZ - pinY);
	btCollisionShape* pin = PhysicsState::engine->createCylinderShape(pinY, pinX, pinY);
	
	btCompoundShape* pinShape = new btCompoundShape();
	const btMatrix3x3 yTonX(0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	btTransform childT(yTonX);
	childT.setOrigin(btVector3(0, 0, pinZ - pinY));
	pinShape->addChildShape(childT, pin);
	childT.setOrigin(btVector3(0, 0, -pinZ + pinY));
	pinShape->addChildShape(childT, pin);
	childT.setIdentity();
	pinShape->addChildShape(childT, pad);
	
	return pinShape;
}

void TrackedPhysics::createChassis(Chassis & c)
{
	const Vector3F dims = c.extends() * .5f;
	btCollisionShape* chassisShape = PhysicsState::engine->createBoxShape(dims.x - 0.1f, dims.y, dims.z);
	
	const Vector3F origin = c.center();
	btTransform trans;
	trans.setIdentity();
	trans.setOrigin(btVector3(origin.x, origin.y, origin.z));
	btRigidBody* chassisBody = PhysicsState::engine->createRigitBody(chassisShape, trans, 10.f);
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
	btCollisionShape* wheelShape = simpleWheelShape(profile);
	
	createWheel(wheelShape, profile);
}

void TrackedPhysics::createCompoundWheel(CreateWheelProfile & profile)
{
	btCollisionShape* wheelShape = compoundWheelShape(profile);
	
	createWheel(wheelShape, profile);
}

void TrackedPhysics::createWheel(btCollisionShape* wheelShape, CreateWheelProfile & profile)
{
	const btMatrix3x3 yTonX(0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	const btMatrix3x3 yToX(0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	
	btTransform trans(yTonX);
	if(!profile.isLeft) trans = btTransform(yToX);
	trans.setOrigin(btVector3(profile.worldP.x, profile.worldP.y, profile.worldP.z));
	btRigidBody* wheelBody = PhysicsState::engine->createRigitBody(wheelShape, trans, profile.mass);
	wheelBody->setDamping(.0f, .0f);
	wheelBody->setFriction(CONTACTFRICTION);
	
	const btMatrix3x3 zToX(0.f, 0.f, -1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f);
	const btMatrix3x3 zTonX(0.f, 0.f, 1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 0.f);
	
	btTransform frameInA(zTonX);
	if(!profile.isLeft) frameInA = btTransform(zToX);
	
	frameInA.setOrigin(btVector3(profile.objectP.x, profile.objectP.y, profile.objectP.z));
	
	const btMatrix3x3 zTonY(1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, -1.f, 0.f);
	
	btTransform frameInB(zTonY);
	btGeneric6DofConstraint* hinge = PhysicsState::engine->constrainByHinge(*profile.connectTo, *wheelBody, frameInA, frameInB, true);
	profile.dstBody = wheelBody;
	profile.dstHinge = hinge;
}

btCollisionShape* TrackedPhysics::simpleWheelShape(CreateWheelProfile & profile)
{
	btCollisionShape* wheelShape = PhysicsState::engine->createCylinderShape(profile.radius, profile.width * .5f, profile.radius);
	return wheelShape;
}

btCollisionShape* TrackedPhysics::compoundWheelShape(CreateWheelProfile & profile)
{
	float rollWidth = (profile.width - profile.gap) * .5f;
	btCollisionShape* rollShape = PhysicsState::engine->createCylinderShape(profile.radius, rollWidth * .5f, profile.radius);
	btCompoundShape* wheelShape = new btCompoundShape();
	
	btTransform childT; childT.setIdentity();
	childT.setOrigin(btVector3(0, rollWidth * 0.5 + profile.gap * 0.5, 0));
	wheelShape->addChildShape(childT, rollShape);
	childT.setOrigin(btVector3(0, rollWidth * -0.5 - profile.gap * 0.5, 0));
	wheelShape->addChildShape(childT, rollShape);
	return wheelShape;
}
	
btCollisionShape* TrackedPhysics::createSprocketShape(CreateWheelProfile & profile)
{
	float rollWidth = (profile.width - profile.gap) * .5f;
	btCollisionShape* rollShape = PhysicsState::engine->createCylinderShape(profile.radius, rollWidth * .5f, profile.radius);
	btCollisionShape* toothShape = PhysicsState::engine->createCylinderShape(Tread::ToothWidth * 0.4f, Tread::ToothWidth * 0.5f, Tread::ToothWidth * 0.4f);
	
	btCompoundShape* wheelShape = new btCompoundShape();
	
	btTransform childT; childT.setIdentity();
	childT.setOrigin(btVector3(0, rollWidth * 0.5 + profile.gap * 0.5, 0));
	wheelShape->addChildShape(childT, rollShape);
	childT.setOrigin(btVector3(0, rollWidth * -0.5 - profile.gap * 0.5, 0));
	wheelShape->addChildShape(childT, rollShape);
	
	const float toothR = profile.radius + Tread::ToothWidth * 0.5f;
	
	const float delta = PI * 2.f / 11.f;
	for(int i = 0; i < 11; i++) {
		childT.setOrigin(btVector3(cos(delta * i) * toothR, profile.width * 0.5f - Tread::ToothWidth * 0.5f, sin(delta * i) * toothR) );
		wheelShape->addChildShape(childT, toothShape);
		childT.getOrigin()[1] = profile.width * -0.5f + Tread::ToothWidth * 0.5f;
		wheelShape->addChildShape(childT, toothShape);
	}
	
	return wheelShape;
}

void TrackedPhysics::createDriveSprocket(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.driveSprocketRadius();
	cwp.width = c.trackWidth();
	cwp.mass = 2.f;
	cwp.worldP = c.driveSprocketOrigin(isLeft);
	cwp.objectP = c.driveSprocketOriginObject(isLeft);
	cwp.isLeft = isLeft;
	cwp.gap = c.trackWidth() * .5f;
	//createCompoundWheel(cwp);
	btCollisionShape* sprocketShape = createSprocketShape(cwp);
	createWheel(sprocketShape, cwp);
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
	cwp.mass = 2.f;
	cwp.worldP = c.tensionerOrigin(isLeft);
	cwp.objectP = c.tensionerOriginObject(isLeft);
	cwp.isLeft = isLeft;
	cwp.gap = Tread::ToothWidth * 1.1f;
	createCompoundWheel(cwp);
	if(isLeft) m_tension[0] = cwp.dstHinge;
	else m_tension[1] = cwp.dstHinge;
	
}

void TrackedPhysics::createRoadWheels(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	if(c.numRoadWheels() < 1) return;
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.roadWheelRadius();
	cwp.width = c.trackWidth() * .5f;
	cwp.mass = 2.f;
	cwp.isLeft = isLeft;
	cwp.gap = Tread::ToothWidth * 1.1f;
	for(int i=0; i < c.numRoadWheels(); i++) {
		btRigidBody * torsionBar = createTorsionBar(chassisBody, i, isLeft);
		Vector3F p = roadWheelOrigin(i, isLeft);
		
		cwp.connectTo = torsionBar;
		cwp.worldP = p;
		cwp.objectP = p - torsionBarHinge(i, isLeft) + Vector3F::ZAxis * torsionBarLength() * .5;
		createCompoundWheel(cwp);
	}
}

void TrackedPhysics::createSupportRollers(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	if(c.numSupportRollers() < 1) return;
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.supportRollerRadius();
	cwp.width = c.trackWidth() * .5f;
	cwp.mass = 1.f;
	cwp.isLeft = isLeft;
	cwp.gap = Tread::ToothWidth * 1.1f;
	for(int i=0; i < c.numSupportRollers(); i++) {
		cwp.worldP = c.supportRollerOrigin(i, isLeft);
		cwp.objectP = c.supportRollerOriginObject(i, isLeft);
		createCompoundWheel(cwp);
	}
}

void TrackedPhysics::addTension(const float & x)
{
	btGeneric6DofConstraint* te = m_tension[0];
	btTransform & frm = te->getFrameOffsetA();
	btVector3 & p = frm.getOrigin();
	if(p[2] < 43)p[2] += x * .1;
	
	te = m_tension[1];
	btTransform & frm1 = te->getFrameOffsetA();
	btVector3 & p1 = frm1.getOrigin();
	if(p[2] < 43)p1[2] += x * .1;
}

void TrackedPhysics::addPower(const float & x)
{
	m_targeVelocity += x;
	
	m_drive[0]->getRotationalLimitMotor(2)->m_enableMotor = true;
	m_drive[0]->getRotationalLimitMotor(2)->m_targetVelocity = -m_targeVelocity;
	if(m_drive[0]->getRotationalLimitMotor(2)->m_maxMotorForce < 10000.f )
		m_drive[0]->getRotationalLimitMotor(2)->m_maxMotorForce += 100.f;
	m_drive[0]->getRotationalLimitMotor(2)->m_damping = 0.5f;
	m_drive[1]->getRotationalLimitMotor(2)->m_enableMotor = true;
	m_drive[1]->getRotationalLimitMotor(2)->m_targetVelocity  = m_targeVelocity;
	if(m_drive[1]->getRotationalLimitMotor(2)->m_maxMotorForce < 10000.f )
		m_drive[1]->getRotationalLimitMotor(2)->m_maxMotorForce += 100.f;
	m_drive[1]->getRotationalLimitMotor(2)->m_damping = 0.5f;
}

void TrackedPhysics::addBrake(bool leftSide)
{
	if(leftSide) {
		m_drive[0]->getRotationalLimitMotor(2)->m_targetVelocity = 0.;
		m_drive[0]->getRotationalLimitMotor(2)->m_maxMotorForce = 1000.f;
	}
	else {
		m_drive[1]->getRotationalLimitMotor(2)->m_targetVelocity = 0.;
		m_drive[1]->getRotationalLimitMotor(2)->m_maxMotorForce = 1000.f;
	}
}

btRigidBody * TrackedPhysics::createTorsionBar(btRigidBody * chassisBody, const int & i, bool isLeft)
{
	btCollisionShape* torsionBarShape = PhysicsState::engine->createBoxShape(torsionBarSize() * .5f, torsionBarSize() * .5f, torsionBarLength() * .5f);
	btTransform trans;
	trans.setIdentity();
	Vector3F p = torsionBarHinge(i, isLeft);
	p.z -= torsionBarLength() * .5f;
	trans.setOrigin(btVector3(p.x, p.y, p.z));
	btRigidBody * body = PhysicsState::engine->createRigitBody(torsionBarShape, trans, 1.f);
	body->setDamping(0., 1.);
	
	p = torsionBarHingeObject(i, isLeft);
	
	const btMatrix3x3 zToX(0.f, 0.f, -1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f);
	const btMatrix3x3 zTonX(0.f, 0.f, 1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 0.f);
	
	btTransform frameInA(zTonX);
	if(!isLeft) frameInA.setBasis(zToX);
	frameInA.setOrigin(btVector3(p.x, p.y, p.z));
	
	btTransform frameInB(zTonX);
	if(!isLeft) frameInB.setBasis(zToX);
	frameInB.setOrigin(btVector3(0, 0, torsionBarLength() * .5f));
	
	btGeneric6DofSpringConstraint* spring = PhysicsState::engine->constrainBySpring(*chassisBody, *body, frameInA, frameInB, true);
	spring->setLinearUpperLimit(btVector3(0., 0., 0.));
	spring->setLinearLowerLimit(btVector3(0., 0., 0.));
	
	if(isLeft) {
		spring->setAngularLowerLimit(btVector3(0.f, 0.f, 0.f));
		spring->setAngularUpperLimit(btVector3(0.f, 0.f, 1.f));
	}
	else {
		spring->setAngularLowerLimit(btVector3(0.f, 0.f, -1.f));
		spring->setAngularUpperLimit(btVector3(0.f, 0.f, 0.f));
	}
	
	spring->enableSpring(5, true);
	spring->setStiffness(5, 8000.);
	spring->setDamping(0., 0.5);
	if(isLeft)
		spring->setEquilibriumPoint(5, 0.5);
	else
		spring->setEquilibriumPoint(5, -0.5);
	return body;
}
}
