/*
 *  dynamicsSolver.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "DynamicsSolver.h"
#include "btBulletDynamicsCommon.h"
#include "btBulletCollisionCommon.h"
#include "BulletSoftBody/btSoftBody.h"
#include "BulletSoftBody/btSoftRigidDynamicsWorld.h"
#include "BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h"
#include "BulletSoftBody/btSoftBodyHelpers.h"

DynamicsSolver::DynamicsSolver() : m_enablePhysics(true), m_numSubSteps(2)
{
	m_isWorldInitialized = false;
    _drawer = new ShapeDrawer();
}

DynamicsSolver::~DynamicsSolver()
{
    killPhysics();
}

void DynamicsSolver::setEnablePhysics(bool x) { m_enablePhysics = x; }
void DynamicsSolver::setNumSubSteps(int x) { m_numSubSteps = x; }
const bool DynamicsSolver::isPhysicsEnabled() const { return m_enablePhysics; }
	
void DynamicsSolver::initPhysics()
{
    // m_collisionConfiguration = new btSoftBodyRigidBodyCollisionConfiguration();
    m_collisionConfiguration = new btDefaultCollisionConfiguration();
	m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);
	m_overlappingPairCache = new btDbvtBroadphase();
	m_constraintSolver = new btSequentialImpulseConstraintSolver();
	m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher, m_overlappingPairCache, m_constraintSolver,m_collisionConfiguration);
	//m_dynamicsWorld->setDebugDrawer(&gDebugDraw);
	
	m_dynamicsWorld->setGravity(btVector3(0,-9.8,0));
	
/*
	btSoftBodyWorldInfo worldInfo;
	worldInfo.m_dispatcher = m_dispatcher;

	btVector3 worldMin(-1000,-1000,-1000);
	btVector3 worldMax(1000,1000,1000);
	
	m_broadphase = new btAxisSweep3(worldMin,worldMax, 1000);

	worldInfo.m_broadphase = m_broadphase;
	worldInfo.m_sparsesdf.Initialize();
    worldInfo.m_gravity.setValue(0,0,0);
	m_constraintSolver = new btSequentialImpulseConstraintSolver();
	m_dynamicsWorld = new btSoftRigidDynamicsWorld(m_dispatcher, m_broadphase, m_constraintSolver, m_collisionConfiguration);

	m_dynamicsWorld->getDispatchInfo().m_enableSPU = true;

	m_dynamicsWorld->setGravity(btVector3(0,-1,0));
*/

	clientBuildPhysics();
	
	/*
	btCollisionShape* cubeShape2 = new btBoxShape(btVector3(2.f,.2f,.5f));
	m_collisionShapes.push_back(cubeShape2);
	
	trans.setOrigin(btVector3(5.0, 9.0, 3.0));
	btRigidBody* clavicle = localCreateRigidBody(1.f, trans, cubeShape2);
	m_dynamicsWorld->addRigidBody(clavicle);
	
	btCollisionShape* cubeShape3 = new btBoxShape(btVector3(4.f,.2f,.5f));
	m_collisionShapes.push_back(cubeShape3);
	
	trans.setOrigin(btVector3(20.0, 9.0, 4.0));
	btRigidBody* body2 = localCreateRigidBody(1.f, trans, cubeShape3);
	m_dynamicsWorld->addRigidBody(body2);
	
	trans.setOrigin(btVector3(25.0, 9.0, 6.0));
	btRigidBody* body3 = localCreateRigidBody(1.f, trans, cubeShape3);
	m_dynamicsWorld->addRigidBody(body3);
	
	btCollisionShape* cubeShape4 = new btBoxShape(btVector3(.5f,.5f,.5f));
	m_collisionShapes.push_back(cubeShape4);
	
	trans.setOrigin(btVector3(28.0, 9.0, 12.0));
	btRigidBody* body4 = localCreateRigidBody(1.f, trans, cubeShape4);
	m_dynamicsWorld->addRigidBody(body4);
	
	btMatrix3x3 flip(1.f, 0.f, 0.f, 0.f, 0.f, -1.f, 0.f, 1.f, 0.f);
	btTransform frameInA(flip), frameInB(flip);
    
    frameInA.setOrigin(btVector3(2., 4., 0.));
    frameInB.setOrigin(btVector3(-2.5, 0., 0.));
	btGeneric6DofConstraint* d6f = new btGeneric6DofConstraint(*body0, *clavicle, frameInA, frameInB, true);
	d6f->setAngularLowerLimit(btVector3(0., -SIMD_PI/4., -SIMD_PI/4.));
    d6f->setAngularUpperLimit(btVector3(0., SIMD_PI/4., SIMD_PI/4.));	
	m_dynamicsWorld->addConstraint(d6f);
	
	frameInA.setOrigin(btVector3(2.5, 0., 0.));
    frameInB.setOrigin(btVector3(-6., 0., 0.));
	
	btGeneric6DofConstraint* d6f1 = new btGeneric6DofConstraint(*body2, *clavicle, frameInB, frameInA, true);
	d6f1->setAngularLowerLimit(btVector3(-SIMD_PI/2.3, -SIMD_PI/2.1, -SIMD_PI/22.3));
    d6f1->setAngularUpperLimit(btVector3(SIMD_PI/2.3, SIMD_PI/12.3, SIMD_PI/1.8));	
	m_dynamicsWorld->addConstraint(d6f1);
	
	frameInA.setOrigin(btVector3(6., 0., 0.));
    frameInB.setOrigin(btVector3(0., 0., 0.));
	
	btGeneric6DofConstraint* d6f2 = new btGeneric6DofConstraint(*body2, *body4, frameInA, frameInB, true);
	//d6f2->setAngularLowerLimit(btVector3(0., 0., -SIMD_PI* .75));
    //d6f2->setAngularUpperLimit(btVector3(0., 0., 0.));
    //d6f2->setAngularLowerLimit(btVector3(0., 0., 0.));
    //d6f2->setAngularUpperLimit(btVector3(0., 0., 0.));
    d6f2->setLinearLowerLimit(btVector3(-33.3, 0., 0.));
    d6f2->setLinearUpperLimit(btVector3(33.3, 0., 0.));	
	m_dynamicsWorld->addConstraint(d6f2);
	
	frameInA.setOrigin(btVector3(6., 0., 0.));
    frameInB.setOrigin(btVector3(-2., 0., 0.));
	
	btGeneric6DofConstraint* d6f3 = new btGeneric6DofConstraint(*body3, *body4, frameInA, frameInB, true);	
	m_dynamicsWorld->addConstraint(d6f3);
	
	body0->setDamping(.99f, .99f);
	clavicle->setDamping(.99f, .99f);
	body2->setDamping(.99f, .99f);
	body3->setDamping(.99f, .99f);
	body4->setDamping(.99f, .99f);
	
	btCollisionShape* scapulaShape = new btBoxShape(btVector3(1.f,1.5f,.25f));
	m_collisionShapes.push_back(scapulaShape);
	
	trans.setOrigin(btVector3(6.0, 7.0, 4.0));
	btRigidBody* scapula = localCreateRigidBody(1.f, trans, scapulaShape);
	m_dynamicsWorld->addRigidBody(scapula);
	scapula->setDamping(.99f, .99f);
	
	frameInA.setOrigin(btVector3(2.5, 0., 0.));
    frameInB.setOrigin(btVector3(1.2, 1.99, 0.5));
	
	btGeneric6DofConstraint* c2s = new btGeneric6DofConstraint(*clavicle, *scapula, frameInA, frameInB, true);
	c2s->setAngularLowerLimit(btVector3(-SIMD_PI/2.3, -SIMD_PI/2.1, -SIMD_PI/22.3));
    c2s->setAngularUpperLimit(btVector3(SIMD_PI/2.3, SIMD_PI/12.3, SIMD_PI/1.8));	
	m_dynamicsWorld->addConstraint(c2s);
	
	frameInA.setOrigin(btVector3(-1.2, 1.99, 0.));
	frameInB.setOrigin(btVector3(3., 2., -3.));
    btGeneric6DofConstraint* r2s = new btGeneric6DofConstraint(*scapula, *body0, frameInA, frameInB, true);
	r2s->setLinearLowerLimit(btVector3(-3.3, -3.1, -3.3));
    r2s->setLinearUpperLimit(btVector3(3.3, 3.3, 3.8));	
	
    
    m_dynamicsWorld->addConstraint(r2s);
    
    frameInA.setOrigin(btVector3(-1.2, -1.99, 0.));
	frameInB.setOrigin(btVector3(3., -2., -3.));
    btGeneric6DofConstraint* r2s2 = new btGeneric6DofConstraint(*scapula, *body0, frameInA, frameInB, true);
	r2s2->setLinearLowerLimit(btVector3(-3.3, -3.1, -3.3));
    r2s2->setLinearUpperLimit(btVector3(3.3, 3.3, 3.8));	
	
    
    m_dynamicsWorld->addConstraint(r2s2);
	
	*/
	
	m_isWorldInitialized = true;;
}

const bool DynamicsSolver::isWorldInitialized() const
{
	return m_isWorldInitialized;
}

void DynamicsSolver::killPhysics()
{
	if(!m_isWorldInitialized) return;
	//remove the rigidbodies from the dynamics world and delete them
	int i;
	for (i=m_dynamicsWorld->getNumCollisionObjects()-1; i>=0 ;i--)
	{
		btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[i];
		btRigidBody* body = btRigidBody::upcast(obj);
		if (body && body->getMotionState())
		{

			while (body->getNumConstraintRefs())
			{
				btTypedConstraint* constraint = body->getConstraintRef(0);
				m_dynamicsWorld->removeConstraint(constraint);
				delete constraint;
			}
			delete body->getMotionState();
			m_dynamicsWorld->removeRigidBody(body);
		} 
		else
		{
			m_dynamicsWorld->removeCollisionObject( obj );
		}
		delete obj;
	}
	
	for (int j=0;j<m_collisionShapes.size();j++)
	{
		btCollisionShape* shape = m_collisionShapes[j];
		delete shape;
	}
	m_collisionShapes.clear();
	
	delete m_dynamicsWorld;
	delete m_constraintSolver;
	delete m_overlappingPairCache;
	delete m_dispatcher;
	delete m_collisionConfiguration;
	m_isWorldInitialized = false;
}

const int DynamicsSolver::numCollisionObjects() const
{
	if(!isWorldInitialized()) return 0;
	return m_dynamicsWorld->getNumCollisionObjects();
}

btCollisionObject * DynamicsSolver::getCollisionObject(const int & i) const
{
	if(!isWorldInitialized()) return NULL;
	return m_dynamicsWorld->getCollisionObjectArray()[i];
}

btRigidBody* DynamicsSolver::getRigidBody(const int & i) const
{
    btCollisionObject * obj = getCollisionObject(i);
    if(!obj) return NULL;
    return static_cast<btRigidBody *>(obj);
}

void DynamicsSolver::renderWorld()
{
	if(!isWorldInitialized()) return;
	
	_drawer->drawGravity(m_dynamicsWorld->getGravity());
	
	const int	numObjects= m_dynamicsWorld->getNumCollisionObjects();
	btVector3 wireColor(1,0,0);
	for(int i=0;i<numObjects;i++)
	{
		btCollisionObject*	colObj= m_dynamicsWorld->getCollisionObjectArray()[i];
		_drawer->drawObject(colObj);
	}
	/*
	for (  int i=0;i<m_dynamicsWorld->getSoftBodyArray().size();i++) {
		//btSoftBody*	psb=(btSoftBody*)m_dynamicsWorld->getSoftBodyArray()[i];

			//btSoftBodyHelpers::DrawFrame(psb,m_dynamicsWorld->getDebugDrawer());
			//btSoftBodyHelpers::Draw(psb,m_dynamicsWorld->getDebugDrawer(),m_dynamicsWorld->getDrawFlags());
	}
	*/
	const int numConstraints = m_dynamicsWorld->getNumConstraints();
	for(int i=0;i< numConstraints;i++) {
	    btTypedConstraint* constraint = m_dynamicsWorld->getConstraint(i);
	    _drawer->drawConstraint(constraint);
	}
}

void DynamicsSolver::simulate()
{
	if(!m_enablePhysics) return;
	btScalar dt = (btScalar)_clock.getTimeMicroseconds() / 1000000.f; // std::cout<<"dt "<<dt;
	_clock.reset();
	simulate(dt, m_numSubSteps, 150.f);
}

void DynamicsSolver::simulate(const float & dt, const int & numSubsteps, const float & frequency)
{
	m_dynamicsWorld->stepSimulation(dt, numSubsteps, 1.f / frequency);
}

btRigidBody* DynamicsSolver::internalCreateRigidBody(float mass, const btTransform& startTransform, btCollisionShape* shape)
{
	btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0,0,0);
	if (isDynamic)
		shape->calculateLocalInertia(mass,localInertia);
	
	// printf("inertial %f %f %f \n", localInertia.getX(), localInertia.getY(), localInertia.getZ());

	//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
	btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass,myMotionState,shape,localInertia);
	
	btRigidBody* body = new btRigidBody(cInfo);
	body->setContactProcessingThreshold(_defaultContactProcessingThreshold);

	return body;
}

void DynamicsSolver::addGroundPlane(const float & groundSize, const float & groundLevel)
{
	btBoxShape* groundShape = createBoxShape(groundSize, 1.f, groundSize);
	
	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(btVector3(0, groundLevel - 1.f, 0));
	btRigidBody* body = createRigidBody(groundShape,tr,0.f);
	body->setFriction(.9);
}

btBoxShape* DynamicsSolver::createBoxShape(const float & x, const float & y, const float & z)
{
	btBoxShape* cubeShape = new btBoxShape(btVector3(x, y, z));
	m_collisionShapes.push_back(cubeShape);
	return cubeShape;
}

btCylinderShape* DynamicsSolver::createCylinderShape(const float & x, const float & y, const float & z)
{
	btCylinderShape* cyl = new btCylinderShape(btVector3(x, y, z));
	m_collisionShapes.push_back(cyl);
	return cyl;
}

btSphereShape* DynamicsSolver::createSphereShape(const float & r)
{
	btSphereShape* spr = new btSphereShape(r);
	m_collisionShapes.push_back(spr);
	return spr;
}

void DynamicsSolver::addCollisionShape(btCollisionShape* shape)
{
    m_collisionShapes.push_back(shape);
}

btRigidBody* DynamicsSolver::createRigidBody(btCollisionShape* shape, const btTransform & transform, const float & mass)
{
	btRigidBody* body = internalCreateRigidBody(mass, transform, shape);
	m_dynamicsWorld->addRigidBody(body);
	return body;
}

void DynamicsSolver::clientBuildPhysics() {}

btGeneric6DofConstraint* DynamicsSolver::constrainByHinge(btRigidBody& rbA, btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool disableCollisionsBetweenLinkedBodies)
{
	//btHingeConstraint* hinge = new btHingeConstraint(rbA, rbB, rbAFrame, rbBFrame);
	
	btGeneric6DofConstraint* hinge = new btGeneric6DofConstraint(rbA, rbB, rbAFrame, rbBFrame, false);
	hinge->setAngularLowerLimit(btVector3(0.0, 0.0, -SIMD_PI));
    hinge->setAngularUpperLimit(btVector3(0.0, 0.0, SIMD_PI));
	hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
    hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	//hinge->getTranslationalLimitMotor()->m_limitSoftness = 0.f;		
	//hinge->getTranslationalLimitMotor()->m_damping = 0.f;		
	//hinge->getTranslationalLimitMotor()->m_currentLimitError[0] = 0.f;
	//hinge->getTranslationalLimitMotor()->m_currentLimitError[1] = 0.f;
	//hinge->getTranslationalLimitMotor()->m_currentLimitError[2] = 0.f;
	
	m_dynamicsWorld->addConstraint(hinge, disableCollisionsBetweenLinkedBodies);
	return hinge;
}

btGeneric6DofSpringConstraint* DynamicsSolver::constrainBySpring(btRigidBody& rbA, btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool disableCollisionsBetweenLinkedBodies)
{
	btGeneric6DofSpringConstraint* spring = new btGeneric6DofSpringConstraint(rbA, rbB, rbAFrame, rbBFrame, false);
	m_dynamicsWorld->addConstraint(spring, disableCollisionsBetweenLinkedBodies);
	return spring;
}

ShapeDrawer* DynamicsSolver::getDrawer() { return _drawer; }

