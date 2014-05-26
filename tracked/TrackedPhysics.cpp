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
#define CONTACTFRICTION .414
#define WHEELMASS 1.0
#define SHOEMASS .5
#define PINMASS .5
TrackedPhysics::TrackedPhysics() 
{ 
	addGroup("chassis");
	addGroup("left_driveSprocket");
	addGroup("right_driveSprocket");
	addGroup("left_tensioner");
	addGroup("right_tensioner");
	addGroup("left_roadWheel");
	addGroup("right_roadWheel");
	addGroup("left_supportRoller");
	addGroup("right_supportRoller");
	addGroup("left_bogieArm");
	addGroup("right_bogieArm");
	addGroup("left_trackShoe");
	addGroup("right_trackShoe");
	addGroup("left_trackPin");
	addGroup("right_trackPin");
	m_targeVelocity = 0.f;
}

TrackedPhysics::~TrackedPhysics() {}

void TrackedPhysics::create()
{
	resetGroups();

	createChassis(*this);
	
	Tread::SprocketRadius = driveSprocketRadius();
	
	m_leftTread.setWidth(trackWidth());
	m_rightTread.setWidth(trackWidth());

	addTreadSections(m_leftTread);
	addTreadSections(m_rightTread, false);
	
	createTread(m_leftTread);
	createTread(m_rightTread, false);
}

void TrackedPhysics::addTreadSections(Tread & t, bool isLeft)
{
	Vector3F p, q;
	p = driveSprocketOrigin(isLeft) - Vector3F::YAxis * driveSprocketRadius();
	q = roadWheelRestPosition(numRoadWheels() - 1, isLeft) - Vector3F::YAxis * roadWheelRadius(); 
	
	Tread::Section sect;
	sect._type = Tread::Section::tLinear;
	sect._initialPosition = p;
	sect._eventualPosition = q;
	const float nb = asin((p.y - q.y) / (p - q).length());
	sect._initialAngle = nb;
	
	t.addSection(sect);
	
	p = q;
	q = roadWheelRestPosition(0, isLeft) - Vector3F::YAxis * roadWheelRadius();
	sect._type = Tread::Section::tLinear;
	sect._initialPosition = p;
	sect._eventualPosition = q;
	sect._initialAngle *= -1.f;
	
	t.addSection(sect);

	p = q;
	q = tensionerOrigin(isLeft) - Vector3F::YAxis * tensionerRadius();
	
	float na = asin((p.y - q.y) / (p - q).length());
	
	sect._type = Tread::Section::tLinear;
	sect._initialPosition = p;
	sect._eventualPosition = q;
	sect._initialAngle = na;
	
	t.addSection(sect);
	
	sect._type = Tread::Section::tAngular;
	sect._rotateAround = tensionerOrigin(isLeft);
	sect._initialAngle = 0.f;
	sect._eventualAngle = - PI - na;
	sect._rotateRadius = tensionerRadius();
	sect._initialPosition = tensionerOrigin(isLeft) - Vector3F::YAxis * tensionerRadius();
	
	t.addSection(sect);
	
	p = tensionerOrigin(isLeft) + Vector3F::YAxis * tensionerRadius();
	q = driveSprocketOrigin(isLeft) + Vector3F::YAxis * driveSprocketRadius();
	sect._type = Tread::Section::tLinear;
	sect._initialAngle = 0.f;
	sect._initialPosition = p;
	sect._eventualPosition = q;
	
	t.addSection(sect);
	
	sect._type = Tread::Section::tAngular;
	sect._rotateAround = driveSprocketOrigin(isLeft);
	sect._initialAngle = - nb * .5f;
	sect._eventualAngle = - PI - na + nb *.5f;
	sect._rotateRadius = driveSprocketRadius();
	sect._initialPosition = driveSprocketOrigin(isLeft) + Vector3F::YAxis * driveSprocketRadius();
	
	t.addSection(sect);
	
	t.computeSections();
}

void TrackedPhysics::createObstacles()
{
	btCollisionShape* obstacleShape = PhysicsState::engine->createBoxShape(40, 1, 4);
	btTransform trans; trans.setIdentity(); trans.setOrigin(btVector3(10,1,50));
	btRigidBody* obs = PhysicsState::engine->createRigidBody(obstacleShape, trans, 0.f);
	obs->setDamping(0,0);
	obs->setFriction(.5);
	trans.setOrigin(btVector3(-10,1,80));
	obs = PhysicsState::engine->createRigidBody(obstacleShape, trans, 0.f);
	obs->setDamping(0,0);
	obs->setFriction(.5);
}

void TrackedPhysics::createTread(Tread & tread, bool isLeft)
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
	
	const float hseg = tread.segLength() * .5f;
	const float hingeFac = tread.pinHingeFactor();
	const float invHingeFac = 1.f - hingeFac;
	
	tread.begin();
	while(!tread.end()) {
		Vector3F at = tread.currentSpace().getTranslation();
		Matrix33F rot = tread.currentSpace().rotation();
		trans = btTransform(btMatrix3x3(rot.M(0, 0), rot.M(1, 0), rot.M(2, 0), rot.M(0, 1), rot.M(1, 1), rot.M(2, 1), rot.M(0, 2), rot.M(1, 2), rot.M(2, 2)));
		trans.setOrigin(btVector3(at.x, at.y, at.z));
		
		const int id = PhysicsState::engine->numCollisionObjects();
		if(tread.currentIsShoe()) {
			curBody = PhysicsState::engine->createRigidBody(shoeShape, trans, SHOEMASS);
			if(isLeft) group("left_trackShoe").push_back(id);
			else group("right_trackShoe").push_back(id);
			
			curBody->setFriction(CONTACTFRICTION);
		}
		else {
			curBody = PhysicsState::engine->createRigidBody(pinShape, trans, PINMASS);
			if(isLeft) group("left_trackPin").push_back(id);
			else group("right_trackPin").push_back(id);
			
			curBody->setFriction(0.);
		}
			
		curBody->setDamping(.09f, .99f);
		
		if(!firstBody) firstBody = curBody;
			
		if(preBody) {
			btTransform frameInA(zToX), frameInB(zToX);
	
			if(tread.currentIsShoe()) {
				frameInA.setOrigin(btVector3(0.0, 0.0, hseg * hingeFac));
				frameInB.setOrigin(btVector3(0.0, shoeY * tread.ShoeHingeRise, hseg * -1.f * invHingeFac));
			}
			else {
				frameInA.setOrigin(btVector3(0.0, shoeY * tread.ShoeHingeRise, hseg * invHingeFac));
				frameInB.setOrigin(btVector3(0.0, 0.0, hseg * -1.f * hingeFac));
			}
			threePointHinge(frameInA, frameInB, tread.width() * 0.5f, preBody, curBody);
		}
		
		preBody = curBody;
		tread.next();
	}
	
	btTransform frameInShoe(zToX), frameInPin(zToX);
	frameInShoe.setOrigin(btVector3(0.f, shoeY * tread.ShoeHingeRise, hseg * -1.f * invHingeFac));
	frameInPin.setOrigin(btVector3(0.f, 0.0, hseg * hingeFac));


	threePointHinge(frameInPin, frameInShoe, tread.width() * 0.5f, curBody, firstBody);
}

void TrackedPhysics::threePointHinge(btTransform & frameInA, btTransform & frameInB, const float & side, btRigidBody* bodyA, btRigidBody* bodyB)
{
	btGeneric6DofConstraint * hinge = PhysicsState::engine->constrainByHinge(*bodyA, *bodyB, frameInA, frameInB, true);
	
	hinge->setAngularUpperLimit(btVector3(0.0, 0.0, 0.00023));
	
	btVector3 & p = frameInA.getOrigin();
	p[0] = -side + Tread::ToothWidth * .5f;
	btVector3 & p1 = frameInB.getOrigin();
	p1[0] = -side + Tread::ToothWidth * .5f;
	
	hinge = PhysicsState::engine->constrainByHinge(*bodyA, *bodyB, frameInA, frameInB, true);

	hinge->setAngularUpperLimit(btVector3(0.0, 0.0, 0.00023));
	
	btVector3 & p2 = frameInA.getOrigin();
	p2[0] = side - Tread::ToothWidth * .5f;
	btVector3 & p3 = frameInB.getOrigin();
	p3[0] = side - Tread::ToothWidth * .5f;
	
	hinge = PhysicsState::engine->constrainByHinge(*bodyA, *bodyB, frameInA, frameInB, true);
	
	hinge->setAngularUpperLimit(btVector3(0.0, 0.0, 0.00023));
}

btCollisionShape* TrackedPhysics::createShoeShape(const float & x, const float &y, const float & z)
{
	btCollisionShape* pad = PhysicsState::engine->createBoxShape(x, y, z);
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
	const float side = tread.width() * .5f - Tread::ToothWidth * .5f;
	const float pinX = Tread::ToothWidth * .5f;
	const float pinZ = (tread.pinLength() - tread.pinThickness())  * 0.5f;
	const float pinY = tread.pinThickness() * .5f;
	
	btCollisionShape* pad = PhysicsState::engine->createBoxShape(pinX, pinY, pinZ);
	btCollisionShape* pin = PhysicsState::engine->createCylinderShape(pinY, pinX, pinY);
	
	btCompoundShape* pinShape = new btCompoundShape();
	const btMatrix3x3 yTonX(0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
	btTransform childT;
	childT.setIdentity();
	
	childT.setOrigin(btVector3(side, 0, 0));
	pinShape->addChildShape(childT, pad);
	
	childT.setOrigin(btVector3(-side, 0, 0));
	pinShape->addChildShape(childT, pad);
	
	childT.setBasis(yTonX);
	childT.setOrigin(btVector3(side, 0, pinZ));
	pinShape->addChildShape(childT, pin);
	
	childT.setOrigin(btVector3(-side, 0, pinZ));
	pinShape->addChildShape(childT, pin);
	
	childT.setOrigin(btVector3(side, 0, -pinZ));
	pinShape->addChildShape(childT, pin);
	childT.setOrigin(btVector3(-side, 0, -pinZ));
	pinShape->addChildShape(childT, pin);
	
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
	
	const int id = PhysicsState::engine->numCollisionObjects();
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(chassisShape, trans, 10.f);
	group("chassis").push_back(id);
	
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
	btRigidBody* wheelBody = PhysicsState::engine->createRigidBody(wheelShape, trans, profile.mass);
	wheelBody->setDamping(0.f, 0.f);
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
	const float side = profile.width * 0.5f - Tread::ToothWidth * 0.4f;
	
	btCollisionShape* rollShape = PhysicsState::engine->createCylinderShape(profile.radius, rollWidth * .5f, profile.radius);
	btCollisionShape* tooth1Shape = PhysicsState::engine->createCylinderShape(Tread::ToothWidth * 0.5f, Tread::ToothWidth * 0.5f, Tread::ToothWidth * 0.5f );
	//btCollisionShape* tooth2Shape = PhysicsState::engine->createCylinderShape(Tread::ToothWidth * 0.25f, Tread::ToothWidth * 0.5f, Tread::ToothWidth * 0.25f );
	
	btCompoundShape* wheelShape = new btCompoundShape();
	
	btTransform childT; childT.setIdentity();
	childT.setOrigin(btVector3(0, profile.width * 0.5f - rollWidth * 0.5f, 0));
	wheelShape->addChildShape(childT, rollShape);
	childT.setOrigin(btVector3(0, -profile.width * 0.5f + rollWidth * 0.5f, 0));
	wheelShape->addChildShape(childT, rollShape);
	
	const float tooth1R = profile.radius + Tread::ToothWidth * 0.1f;
	//const float tooth2R = profile.radius + m_leftTread.pinThickness();
	const float delta = PI * 2.f / 11.f;
	for(int i = 0; i < 11; i++) {		
		childT.setOrigin(btVector3(cos(delta * i) * tooth1R, side, sin(delta * i) * tooth1R) );
		wheelShape->addChildShape(childT, tooth1Shape);
		childT.getOrigin()[1] = -side;
		wheelShape->addChildShape(childT, tooth1Shape);
		//childT.setOrigin(btVector3(cos(delta * i) * tooth2R, side, sin(delta * i) * tooth2R) );
		//wheelShape->addChildShape(childT, tooth2Shape);
		//childT.getOrigin()[1] = -side;
		//wheelShape->addChildShape(childT, tooth2Shape);
	}
	
	return wheelShape;
}

void TrackedPhysics::createDriveSprocket(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.driveSprocketRadius();
	cwp.width = c.trackWidth();
	cwp.mass = WHEELMASS;
	cwp.worldP = c.driveSprocketOrigin(isLeft);
	cwp.objectP = c.driveSprocketOriginObject(isLeft);
	cwp.isLeft = isLeft;
	cwp.gap = c.trackWidth() * .5f;
	
	btCollisionShape* sprocketShape = createSprocketShape(cwp);
	
	const int id = PhysicsState::engine->numCollisionObjects();
	createWheel(sprocketShape, cwp);
	
	//cwp.dstBody->setFriction(.1f);
	
	if(isLeft) group("left_driveSprocket").push_back(id);
	else group("right_driveSprocket").push_back(id);
	
	btGeneric6DofConstraint* hinge = cwp.dstHinge;
	if(isLeft) m_drive[0] = hinge;
	else m_drive[1] = hinge;
}

void TrackedPhysics::createTensioner(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.tensionerRadius();
	cwp.width = c.trackWidth();
	cwp.mass = WHEELMASS;
	cwp.worldP = c.tensionerOrigin(isLeft);
	cwp.objectP = c.tensionerOriginObject(isLeft);
	cwp.isLeft = isLeft;
	cwp.gap = Tread::ToothWidth * 1.2f;
	const int id = PhysicsState::engine->numCollisionObjects();
	createCompoundWheel(cwp);
	if(isLeft) group("left_tensioner").push_back(id);
	else group("right_tensioner").push_back(id);

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
	cwp.mass = WHEELMASS;
	cwp.isLeft = isLeft;
	cwp.gap = Tread::ToothWidth * 1.2f;
	for(int i=0; i < c.numRoadWheels(); i++) {
		btRigidBody * torsionBar = createTorsionBar(chassisBody, i, isLeft);
		Vector3F p = roadWheelOrigin(i, isLeft);
		
		cwp.connectTo = torsionBar;
		cwp.worldP = p;
		cwp.objectP = p - torsionBarHinge(i, isLeft) + Vector3F::ZAxis * torsionBarLength() * .5;
		
		const int id = PhysicsState::engine->numCollisionObjects();
		createCompoundWheel(cwp);
		if(isLeft) group("left_roadWheel").push_back(id);
		else group("right_roadWheel").push_back(id);
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
	cwp.gap = Tread::ToothWidth * 1.2f;
	for(int i=0; i < c.numSupportRollers(); i++) {
		cwp.worldP = c.supportRollerOrigin(i, isLeft);
		cwp.objectP = c.supportRollerOriginObject(i, isLeft);
		const int id = PhysicsState::engine->numCollisionObjects();
		createCompoundWheel(cwp);
		if(isLeft) group("left_supportRoller").push_back(id);
		else group("right_supportRoller").push_back(id);
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
	//if(m_drive[0]->getRotationalLimitMotor(2)->m_maxMotorForce < 10000.f )
		m_drive[0]->getRotationalLimitMotor(2)->m_maxMotorForce = 100.f;
	m_drive[0]->getRotationalLimitMotor(2)->m_damping = 0.5f;
	m_drive[1]->getRotationalLimitMotor(2)->m_enableMotor = true;
	m_drive[1]->getRotationalLimitMotor(2)->m_targetVelocity  = m_targeVelocity;
	//if(m_drive[1]->getRotationalLimitMotor(2)->m_maxMotorForce < 10000.f )
		m_drive[1]->getRotationalLimitMotor(2)->m_maxMotorForce = 100.f;
	m_drive[1]->getRotationalLimitMotor(2)->m_damping = 0.5f;
}

void TrackedPhysics::addBrake(bool leftSide)
{
	if(leftSide) {
		m_drive[0]->getRotationalLimitMotor(2)->m_targetVelocity = 0.;
		m_drive[0]->getRotationalLimitMotor(2)->m_maxMotorForce = 100.f;
	}
	else {
		m_drive[1]->getRotationalLimitMotor(2)->m_targetVelocity = 0.;
		m_drive[1]->getRotationalLimitMotor(2)->m_maxMotorForce = 100.f;
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
	
	const int id = PhysicsState::engine->numCollisionObjects();
	btRigidBody * body = PhysicsState::engine->createRigidBody(torsionBarShape, trans, 1.f);
	if(isLeft) group("left_bogieArm").push_back(id);
	else group("right_bogieArm").push_back(id);
	
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
	spring->setStiffness(5, 9900.);
	spring->setDamping(0., .8);
	if(isLeft)
		spring->setEquilibriumPoint(5, torsionBarRestAngle());
	else
		spring->setEquilibriumPoint(5, -torsionBarRestAngle());
	return body;
}

void TrackedPhysics::setTargetVelocity(const float & x) { m_targeVelocity = x; }

void TrackedPhysics::displayStatistics() const 
{
	btRigidBody * chassisBody = PhysicsState::engine->getRigidBody(getGroup("chassis")[0]);
	const btVector3 chasisVel = chassisBody->getLinearVelocity(); 
	
	btRigidBody * leftSprocketBody = PhysicsState::engine->getRigidBody(getGroup("left_driveSprocket")[0]);
	const btVector3 leftSprocketVel = leftSprocketBody->getAngularVelocity();

	btRigidBody * rightSprocketBody = PhysicsState::engine->getRigidBody(getGroup("right_driveSprocket")[0]);
	const btVector3 rightSprocketVel = rightSprocketBody->getAngularVelocity();

	std::cout<<"target velocity (sprocked angular / vehicle linear): "<<m_targeVelocity<<" / "<<driveSprocketRadius() * m_targeVelocity<<" \n";
	std::cout<<"sprocket angular velocity (left / right): "<<leftSprocketVel.length()<<" / "<<rightSprocketVel.length()<<" \n";
	std::cout<<"vehicle linear velocity: "<<chasisVel.length()<<" \n";
}

}
