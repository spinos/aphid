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
#define CONTACTFRICTION .576
#define WHEELMASS .5
#define SHOEMASS .2
#define PINMASS .3
#define SPROCKETTEETHPROTRUDE 0.075
#define MINTRACKSTIFFNESS 400

static btTransform CopyFromMatrix44F(const Matrix44F & tm)
{
    const btMatrix3x3 rot(tm.M(0, 0), tm.M(0, 1), tm.M(0, 2), 
                    tm.M(1, 0), tm.M(1, 1), tm.M(1, 2),
                    tm.M(2, 0), tm.M(2, 1), tm.M(2, 2));
    btTransform r;
    r.setBasis(rot);
    const btVector3 pos(tm.M(3, 0), tm.M(3, 1), tm.M(3, 2));
    r.setOrigin(pos);
    return r;
}

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
	m_trackTension = 1000.f;
	m_firstMotion = true;
	m_tension[0] = NULL;
	m_tension[1] = NULL;
	m_drive[0] = NULL;
	m_drive[1] = NULL;
}

TrackedPhysics::~TrackedPhysics() {}

void TrackedPhysics::create()
{
    m_firstMotion = true;
	resetGroups();
	
	Tread::SprocketRadius = driveSprocketRadius();
	Tread::ToothWidth = toothWidth();
	Tread::ToothHeight = toothWidth() * 1.2;
	
	m_leftTread.setWidth(trackWidth());
	m_rightTread.setWidth(trackWidth());

	addTreadSections(m_leftTread);
	addTreadSections(m_rightTread, false);
	
	createTread(m_leftTread);
	createTread(m_rightTread, false);
	
	createChassis(*this);
}

void TrackedPhysics::addTreadSections(Tread & t, bool isLeft)
{
    t.clearSections();
	Vector3F p, q;
	float r;
	getBackWheel(p, r, isLeft);
	
	p = p - Vector3F::YAxis * r;		
	q = roadWheelOrigin(numRoadWheels() - 1, isLeft) - Vector3F::YAxis * roadWheelRadius(); 
	
	const float nb = asin((p.y - q.y) / (p - q).length());
	
	p.z -= sin(nb) * r;
	p.y += (1.f - cos(nb)) * r;
	
	q.z -= sin(nb) * roadWheelRadius(); 
	q.y += (1.f - cos(nb)) * roadWheelRadius();
	
	Tread::Section sect;
	sect._type = Tread::Section::tLinear;
	sect._initialPosition = p;
	sect._eventualPosition = q;
	sect._initialAngle = nb;
	
	t.addSection(sect);
	
	if(nb > .1) {
		p = q;
		q = roadWheelOrigin(numRoadWheels() - 1, isLeft) - Vector3F::YAxis * roadWheelRadius();
		sect._type = Tread::Section::tAngular;
		sect._initialPosition = p;
		sect._eventualPosition = q;
		sect._initialAngle = 0.f;
		sect._eventualAngle = -nb;
		sect._rotateAround = roadWheelOrigin(numRoadWheels() - 1, isLeft);
		sect._rotateRadius = roadWheelRadius();
		
		t.addSection(sect);
	}
	
	p = roadWheelOrigin(numRoadWheels() - 1, isLeft) - Vector3F::YAxis * roadWheelRadius(); 
	q = roadWheelOrigin(0, isLeft) - Vector3F::YAxis * roadWheelRadius();
	sect._type = Tread::Section::tLinear;
	sect._initialPosition = p;
	sect._eventualPosition = q;
	sect._initialAngle *= -1.f;
	
	t.addSection(sect);
	
	p = q;
	
	getFrontWheel(q, r, isLeft);
	q = q - Vector3F::YAxis * r;
	
	const float na = asin((p.y - q.y) / (p - q).length());
	
	q = roadWheelOrigin(0, isLeft) - Vector3F::YAxis * roadWheelRadius();
	q.z -= sin(na) * roadWheelRadius();
	q.y += (1.f - cos(na)) * roadWheelRadius();

	if(na < -.1) {
		sect._type = Tread::Section::tAngular;
		sect._initialPosition = p;
		sect._eventualPosition = q;
		sect._initialAngle = 0.f;
		sect._eventualAngle = na;
		sect._rotateAround = roadWheelOrigin(0, isLeft);
		sect._rotateRadius = roadWheelRadius();
		
		t.addSection(sect);
	}
	
	p = q;
	
	getFrontWheel(q, r, isLeft);
	q = q - Vector3F::YAxis * r;
	q.z -= sin(na) * r;
	q.y += (1.f - cos(na)) * r;
	
	sect._type = Tread::Section::tLinear;
	sect._initialPosition = p;
	sect._eventualPosition = q;
	sect._initialAngle = 0.;
	
	t.addSection(sect);
	
	sect._type = Tread::Section::tAngular;
	sect._initialAngle = 0.f;
	sect._eventualAngle = - PI - na;
	
	getFrontWheel(sect._rotateAround, sect._rotateRadius, isLeft);
	sect._initialPosition = q;
	
	t.addSection(sect);
	
	Vector3F rollerP; float rollerR;
	if(aroundFirstSupportRoller(rollerP, rollerR, isLeft)) {
		getFrontWheel(p, r, isLeft);
		p = p + Vector3F::YAxis * r;
		
        q = rollerP + Vector3F::YAxis * rollerR;
		
        const float angF = asin((p.y - q.y) / (p - q).length());

        sect._type = Tread::Section::tLinear;
        sect._initialAngle = -angF;
        sect._initialPosition = p;
        sect._eventualPosition = q;
        t.addSection(sect);
        
        if(numSupportRollers() > 1) {
            p = q;
            q = supportRollerOrigin(numSupportRollers() - 1, isLeft) + Vector3F::YAxis * supportRollerRadius();
            sect._type = Tread::Section::tLinear;
            sect._initialAngle = angF;
            sect._initialPosition = p;
            sect._eventualPosition = q;
            t.addSection(sect);
        }
        
        p = q;
		getBackWheel(q, r, isLeft);
		q = q + Vector3F::YAxis * r;
	        
	    const float angB = asin((p.y - q.y) / (p - q).length());
	    sect._type = Tread::Section::tLinear;
        sect._initialAngle = -angB;
        sect._initialPosition = p;
        sect._eventualPosition = q;
        t.addSection(sect);
	}
	else {
		getFrontWheel(p, r, isLeft);
		p = p + Vector3F::YAxis * r;
		if(aroundLastSupportRoller(rollerP, rollerR, isLeft)) {
			q = rollerP + Vector3F::YAxis * rollerR;
			sect._type = Tread::Section::tLinear;
			sect._initialAngle = 0.f;
			sect._initialPosition = p;
			sect._eventualPosition = q;
			
			t.addSection(sect);
			
			p = q;
		}
		
		getBackWheel(q, r, isLeft);
		q = q + Vector3F::YAxis * r;
		
		const float angB = asin((p.y - q.y) / (p - q).length());
	    
		sect._type = Tread::Section::tLinear;
		sect._initialAngle = -angB;
		sect._initialPosition = p;
		sect._eventualPosition = q;
		
		t.addSection(sect);
	}
	
	sect._type = Tread::Section::tAngular;
	sect._initialAngle = 0.f;
	sect._eventualAngle = - PI + nb;
	getBackWheel(sect._rotateAround, sect._rotateRadius, isLeft);
	
	t.addSection(sect);
	
	t.computeSections(.7);
}

void TrackedPhysics::createObstacles()
{
	btCollisionShape* obstacleShape = PhysicsState::engine->createBoxShape(40, 2, 4);
	btTransform trans; trans.setIdentity(); trans.setOrigin(btVector3(10,2,50));
	btRigidBody* obs = PhysicsState::engine->createRigidBody(obstacleShape, trans, 0.f);
	obs->setDamping(0,0);
	obs->setFriction(.5);
	trans.setOrigin(btVector3(-10,2,80));
	obs = PhysicsState::engine->createRigidBody(obstacleShape, trans, 0.f);
	obs->setDamping(0,0);
	obs->setFriction(.5);
}

void TrackedPhysics::createTread(Tread & tread, bool isLeft)
{		
	const float shoeY = tread.shoeThickness() * 0.5f;
	btCollisionShape* shoeShape = createShoeShape(tread);
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
	
	hinge->setAngularUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	btVector3 & p = frameInA.getOrigin();
	p[0] = -side + toothWidth() * .5f;
	btVector3 & p1 = frameInB.getOrigin();
	p1[0] = -side + toothWidth() * .5f;
	
	hinge = PhysicsState::engine->constrainByHinge(*bodyA, *bodyB, frameInA, frameInB, true);

	hinge->setAngularUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	btVector3 & p2 = frameInA.getOrigin();
	p2[0] = side - toothWidth() * .5f;
	btVector3 & p3 = frameInB.getOrigin();
	p3[0] = side - toothWidth() * .5f;
	
	hinge = PhysicsState::engine->constrainByHinge(*bodyA, *bodyB, frameInA, frameInB, true);
	
	
	hinge->setAngularUpperLimit(btVector3(0.0, 0.0, 0.0));
}

btCollisionShape* TrackedPhysics::createShoeShape(Tread & tread)
{	
	btCollisionShape* pad = PhysicsState::engine->createBoxShape(tread.padWidth() * 0.5f, tread.shoeThickness() * 0.5f, tread.shoeLength() * 0.5f);
	btCompoundShape* shoeShape = new btCompoundShape();
	
	btTransform childT; childT.setIdentity();
	btVector3 & p = childT.getOrigin();
	p[0] = tread.padX();
	shoeShape->addChildShape(childT, pad);
	p[0] *= -1.;
	shoeShape->addChildShape(childT, pad);
	
	return shoeShape;
}

btCollisionShape* TrackedPhysics::createPinShape(Tread & tread)
{
	const float side = tread.width() * .5f - toothWidth() * .5f;
	const float pinX = toothWidth() * .5f;
	const float pinZ = (tread.pinLength() - tread.pinThickness())  * 0.5f;
	const float pinY = tread.pinThickness() * .5f;
	
	btCollisionShape* pad = PhysicsState::engine->createBoxShape(pinX, pinY, pinZ);
	btCollisionShape* pin = PhysicsState::engine->createCylinderShape(pinY, pinX, pinY);
	btCollisionShape* tooth = PhysicsState::engine->createCylinderShape(toothWidth()* .45f, Tread::ToothHeight * .5f, toothWidth()* .45f);
	
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
	
	childT.setIdentity();
	childT.setOrigin(btVector3(0, tread.pinThickness() * .5f + Tread::ToothHeight * .5f,0));
	
	pinShape->addChildShape(childT, tooth);
	
	return pinShape;
}

void TrackedPhysics::createChassis(Chassis & c)
{
	const Vector3F dims = c.extends() * .5f;
	btCollisionShape* hullShape = PhysicsState::engine->createBoxShape(dims.x - 0.2f, dims.y * .5, dims.z);
	btCollisionShape* guardShape = PhysicsState::engine->createBoxShape(dims.x + trackWidth(), .5f, dims.z);
	
	btCompoundShape* compShape = new btCompoundShape();
	
	btTransform childT; childT.setIdentity();
	compShape->addChildShape(childT, hullShape);
	
	childT.setOrigin(btVector3(0, dims.y - .5, 0));
	compShape->addChildShape(childT, guardShape);
	
	const Vector3F origin = c.center();
	btTransform trans;
	trans.setIdentity();
	trans.setOrigin(btVector3(origin.x, origin.y, origin.z));
	
	const int id = PhysicsState::engine->numCollisionObjects();
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(compShape, trans, 40.f);
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
	if(!profile.isSpringConstraint) {
		btGeneric6DofConstraint* hinge = PhysicsState::engine->constrainByHinge(*profile.connectTo, *wheelBody, frameInA, frameInB, true);
		profile.dstHinge = hinge;
	}
	else {
		btGeneric6DofSpringConstraint* spring = PhysicsState::engine->constrainBySpring(*profile.connectTo, *wheelBody, frameInA, frameInB, true);
		profile.dstSpringHinge = spring;
	}
	profile.dstBody = wheelBody;
	
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
	const float side = profile.width * 0.5f - toothWidth() * 0.4f;
	
	btCollisionShape* rollShape = PhysicsState::engine->createCylinderShape(profile.radius, rollWidth * .5f, profile.radius);
	btCollisionShape* tooth1Shape = PhysicsState::engine->createCylinderShape(toothWidth() * 0.5f, toothWidth() * 0.5f, toothWidth() * 0.5f );

	btCompoundShape* wheelShape = new btCompoundShape();
	
	btTransform childT; childT.setIdentity();
	childT.setOrigin(btVector3(0, profile.width * 0.5f - rollWidth * 0.5f, 0));
	wheelShape->addChildShape(childT, rollShape);
	childT.setOrigin(btVector3(0, -profile.width * 0.5f + rollWidth * 0.5f, 0));
	wheelShape->addChildShape(childT, rollShape);
	
	const float tooth1R = profile.radius + toothWidth() * SPROCKETTEETHPROTRUDE;
	const float delta = PI * 2.f / 11.f;
	for(int i = 0; i < 11; i++) {		
		childT.setOrigin(btVector3(cos(delta * i) * tooth1R, side, sin(delta * i) * tooth1R) );
		wheelShape->addChildShape(childT, tooth1Shape);
		childT.getOrigin()[1] = -side;
		wheelShape->addChildShape(childT, tooth1Shape);
	}
	
	return wheelShape;
}

void TrackedPhysics::createDriveSprocket(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.driveSprocketRadius();
	cwp.width = c.trackWidth();
	cwp.mass = WHEELMASS * 2.;
	cwp.worldP = c.driveSprocketOrigin(isLeft);
	cwp.objectP = c.driveSprocketOriginObject(isLeft);
	cwp.isLeft = isLeft;
	cwp.gap = toothWidth() * 1.1f;
	
	btCollisionShape* sprocketShape = createSprocketShape(cwp);
	
	const int id = PhysicsState::engine->numCollisionObjects();
	createWheel(sprocketShape, cwp);
	
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
	cwp.width = c.tensionerWidth();
	cwp.mass = WHEELMASS;
	cwp.worldP = c.tensionerOrigin(isLeft);
	cwp.objectP = c.tensionerOriginObject(isLeft);
	cwp.isLeft = isLeft;
	cwp.gap = toothWidth() * 1.1f;
	cwp.isSpringConstraint = true;
	const int id = PhysicsState::engine->numCollisionObjects();
	createCompoundWheel(cwp);
	if(isLeft) group("left_tensioner").push_back(id);
	else group("right_tensioner").push_back(id);
	
	btGeneric6DofSpringConstraint* spring = cwp.dstSpringHinge;
	spring->setAngularLowerLimit(btVector3(0.f, 0.f, -PI));
	spring->setAngularUpperLimit(btVector3(0.f, 0.f, PI));

	spring->enableSpring(0, true);
	spring->setStiffness(0, m_trackTension);
	spring->setDamping(0, .5);
	
	const float horl = tensionerRadius() * 1.2;
	spring->setLinearLowerLimit(btVector3(-horl, 0., 0.));
	spring->setLinearUpperLimit(btVector3(horl, 0., 0.));
	if(isLeft)
		spring->setEquilibriumPoint(0, -horl * .5);
	else
		spring->setEquilibriumPoint(0, horl * .5);
		
	if(isLeft) m_tension[0] = spring;
	else m_tension[1] = spring;
	
}

void TrackedPhysics::createRoadWheels(Chassis & c, btRigidBody * chassisBody, bool isLeft)
{
	if(c.numRoadWheels() < 1) return;
	CreateWheelProfile cwp;
	cwp.connectTo = chassisBody;
	cwp.radius = c.roadWheelRadius();
	cwp.width = c.roadWheelWidth();
	cwp.mass = WHEELMASS;
	cwp.isLeft = isLeft;
	cwp.gap = toothWidth() * 1.1f;
	for(int i=0; i < c.numRoadWheels(); i++) {
		btRigidBody * torsionBar = createTorsionBar(chassisBody, i, isLeft);
		Vector3F p = roadWheelOrigin(i, isLeft);
		
		cwp.connectTo = torsionBar;
		cwp.worldP = p;
		cwp.objectP = roadWheelOriginToBogie(isLeft);
		
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
	cwp.width = c.supportRollerWidth();
	cwp.mass = .7f;
	cwp.isLeft = isLeft;
	cwp.gap = toothWidth() * 1.1f;
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
	m_trackTension += x;
	if(m_trackTension < MINTRACKSTIFFNESS) m_trackTension = MINTRACKSTIFFNESS;
	m_tension[0]->setStiffness(0, m_trackTension);
	m_tension[1]->setStiffness(0, m_trackTension);
}

void TrackedPhysics::addPower(const float & x)
{
	if(!m_drive[0] || !m_drive[1]) return;
	m_targeVelocity += x;
	
	m_drive[0]->getRotationalLimitMotor(2)->m_enableMotor = true;
	m_drive[0]->getRotationalLimitMotor(2)->m_targetVelocity = -m_targeVelocity;
	m_drive[0]->getRotationalLimitMotor(2)->m_maxMotorForce = 100.f;
	m_drive[0]->getRotationalLimitMotor(2)->m_damping = 0.5f;
	m_drive[1]->getRotationalLimitMotor(2)->m_enableMotor = true;
	m_drive[1]->getRotationalLimitMotor(2)->m_targetVelocity  = m_targeVelocity;
	m_drive[1]->getRotationalLimitMotor(2)->m_maxMotorForce = 100.f;
	m_drive[1]->getRotationalLimitMotor(2)->m_damping = 0.5f;
}

void TrackedPhysics::addBrake(bool leftSide)
{
	if(!m_drive[0] || !m_drive[1]) return;
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
    btCollisionShape* torsionBarShape = PhysicsState::engine->createBoxShape(bogieArmWidth() * .5f, bogieArmWidth() * .5f, bogieArmLength() * .5f);
	const Matrix44F tm = bogieArmOrigin(i, isLeft);
	btTransform trans = CopyFromMatrix44F(tm);

	const int id = PhysicsState::engine->numCollisionObjects();
	btRigidBody * body = PhysicsState::engine->createRigidBody(torsionBarShape, trans, 1.f);
	if(isLeft) group("left_bogieArm").push_back(id);
	else group("right_bogieArm").push_back(id);
	
	body->setDamping(0., 1.);
	
	const btMatrix3x3 zToX(0.f, 0.f, -1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f);
	const btMatrix3x3 zTonX(0.f, 0.f, 1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 0.f);
	
	btTransform frameInA(zTonX);
	if(!isLeft) frameInA.setBasis(zToX);
	
	const Vector3F p = torsionBarHingeObject(i, isLeft);
	frameInA.setOrigin(btVector3(p.x, p.y, p.z));
	
	btTransform frameInB(zTonX);
	if(!isLeft) frameInB.setBasis(zToX);
	frameInB.setOrigin(btVector3(0., 0., .5 * bogieArmLength()));
	
	btGeneric6DofSpringConstraint* spring = PhysicsState::engine->constrainBySpring(*chassisBody, *body, frameInA, frameInB, true);
	spring->setLinearUpperLimit(btVector3(0., 0., 0.));
	spring->setLinearLowerLimit(btVector3(0., 0., 0.));

	spring->enableSpring(5, true);
	spring->setStiffness(5, 4000.);
	spring->setDamping(5, .5);
	
	const float tgt = torsionBarTargetAngle();
	if(isLeft) {
		spring->setAngularLowerLimit(btVector3(0.f, 0.f, tgt - .5f));
	    spring->setAngularUpperLimit(btVector3(0.f, 0.f, tgt + .5f));
	    spring->setEquilibriumPoint(5, tgt);
	}
	else {
		spring->setAngularLowerLimit(btVector3(0.f, 0.f, -tgt - .5f));
	    spring->setAngularUpperLimit(btVector3(0.f, 0.f, -tgt + .5f));
	    spring->setEquilibriumPoint(5, -tgt);
	}
	return body;
}

void TrackedPhysics::setTargetVelocity(const float & x) { m_targeVelocity = x; }

void TrackedPhysics::displayStatistics() const 
{
    if(!PhysicsState::engine->isPhysicsEnabled()) return;
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

const float TrackedPhysics::vehicleSpeed() const
{
	if(!PhysicsState::engine->isPhysicsEnabled()) return 0.f;
	btRigidBody * chassisBody = PhysicsState::engine->getRigidBody(getGroup("chassis")[0]);
	const btVector3 chasisVel = chassisBody->getLinearVelocity(); 
	return chasisVel.length();
}

void TrackedPhysics::setTrackThickness(const float & x) 
{ 
    m_leftTread.setThickness(x);
    m_rightTread.setThickness(x);
}

void TrackedPhysics::setTargetSpeed(const float & lft, const float & rgt)
{
	if(!m_drive[0] || !m_drive[1]) return;
    if(m_firstMotion && lft == 0. && rgt == 0.) return;
    m_firstMotion = false;
    const float lftRps = lft / driveSprocketRadius();
    const float rgtRps = rgt / driveSprocketRadius();
    
    m_drive[0]->getRotationalLimitMotor(2)->m_enableMotor = true;
	m_drive[0]->getRotationalLimitMotor(2)->m_targetVelocity = -lftRps;
	m_drive[0]->getRotationalLimitMotor(2)->m_maxMotorForce = 100.f;
	m_drive[0]->getRotationalLimitMotor(2)->m_damping = 0.5f;
	
	m_drive[1]->getRotationalLimitMotor(2)->m_enableMotor = true;
	m_drive[1]->getRotationalLimitMotor(2)->m_targetVelocity  = rgtRps;
	m_drive[1]->getRotationalLimitMotor(2)->m_maxMotorForce = 100.f;
	m_drive[1]->getRotationalLimitMotor(2)->m_damping = 0.5f;
}

const float TrackedPhysics::trackTension() const { return m_trackTension; }
void TrackedPhysics::setTrackTension(const float & x)
{
	m_trackTension = x;
	if(m_trackTension < MINTRACKSTIFFNESS) m_trackTension = MINTRACKSTIFFNESS;
	if(m_tension[0]) m_tension[0]->setStiffness(0, m_trackTension);
	if(m_tension[1]) m_tension[1]->setStiffness(0, m_trackTension);
}

}
