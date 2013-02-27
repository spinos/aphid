/*
 *  dynamicsSolver.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "dynamicsSolver.h"

void DynamicsSolver::initPhysics()
{
	_defaultContactProcessingThreshold = BT_LARGE_FLOAT;
	
	_collisionConfiguration = new btDefaultCollisionConfiguration();
	_dispatcher = new btCollisionDispatcher(_collisionConfiguration);
	btVector3 worldMin(-1000,-1000,-1000);
	btVector3 worldMax(1000,1000,1000);
	_overlappingPairCache = new btAxisSweep3(worldMin,worldMax);
	_constraintSolver = new btSequentialImpulseConstraintSolver();
	btBroadphaseInterface* broadphase = new btDbvtBroadphase();
	_dynamicsWorld = new btDiscreteDynamicsWorld(_dispatcher, broadphase, _constraintSolver, _collisionConfiguration);
	_dynamicsWorld->setGravity(btVector3(0,0,0));

	btCollisionShape* groundShape = new btBoxShape(btVector3(75,1,75));
	_collisionShapes.push_back(groundShape);
	
	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(btVector3(0,-5,0));
	btRigidBody* body = localCreateRigidBody(0.f,tr,groundShape);


	//_dynamicsWorld->addRigidBody(body);
	
	btCollisionShape* cubeShape = new btBoxShape(btVector3(1.f,1.f,1.f));
	_collisionShapes.push_back(cubeShape);
	
	btTransform trans;
	trans.setIdentity();
	trans.setOrigin(btVector3(0.0, 5.0, 3.0));
	
	btCollisionShape* cubeShape1 = new btBoxShape(btVector3(1.f,4.f,.2f));
	_collisionShapes.push_back(cubeShape1);
	
	btRigidBody* body0 = localCreateRigidBody(1.f, trans, cubeShape1);
	_dynamicsWorld->addRigidBody(body0);
	
	btCollisionShape* cubeShape2 = new btBoxShape(btVector3(3.f,.2f,.2f));
	_collisionShapes.push_back(cubeShape2);
	
	trans.setOrigin(btVector3(5.0, 9.0, 3.0));
	btRigidBody* body1 = localCreateRigidBody(1.f, trans, cubeShape2);
	_dynamicsWorld->addRigidBody(body1);
	
	btCollisionShape* cubeShape3 = new btBoxShape(btVector3(4.f,.2f,.2f));
	_collisionShapes.push_back(cubeShape3);
	
	trans.setOrigin(btVector3(15.0, 9.0, 4.0));
	btRigidBody* body2 = localCreateRigidBody(1.f, trans, cubeShape3);
	_dynamicsWorld->addRigidBody(body2);
	
	trans.setOrigin(btVector3(27.0, 9.0, 6.0));
	btRigidBody* body3 = localCreateRigidBody(1.f, trans, cubeShape3);
	_dynamicsWorld->addRigidBody(body3);
	
	btCollisionShape* cubeShape4 = new btBoxShape(btVector3(.2f,.2f,.2f));
	_collisionShapes.push_back(cubeShape4);
	
	trans.setOrigin(btVector3(35.0, 9.0, 12.0));
	btRigidBody* body4 = localCreateRigidBody(1.f, trans, cubeShape4);
	_dynamicsWorld->addRigidBody(body4);
	
	btTransform frameInA, frameInB;
    frameInA = btTransform::getIdentity();
    frameInB = btTransform::getIdentity();
    frameInA.setOrigin(btVector3(1., 4., 0.));
    frameInB.setOrigin(btVector3(-4., 0., 0.));
	btGeneric6DofConstraint* d6f = new btGeneric6DofConstraint(*body0, *body1, frameInA, frameInB, true);
	d6f->setAngularLowerLimit(btVector3(0., -SIMD_PI/10., -SIMD_PI/20.0));
    d6f->setAngularUpperLimit(btVector3(0., SIMD_PI/10., SIMD_PI/20.0));	
	_dynamicsWorld->addConstraint(d6f);
	
	frameInA.setOrigin(btVector3(4., 0., 0.));
    frameInB.setOrigin(btVector3(-6., 0., 0.));
	
	btGeneric6DofConstraint* d6f1 = new btGeneric6DofConstraint(*body1, *body2, frameInA, frameInB, true);
	d6f1->setAngularLowerLimit(btVector3(0.0, -SIMD_PI/2., -SIMD_PI));
    d6f1->setAngularUpperLimit(btVector3(SIMD_PI/2, SIMD_PI/2., SIMD_PI));	
	_dynamicsWorld->addConstraint(d6f1);
	
	frameInA.setOrigin(btVector3(6., 0., 0.));
    frameInB.setOrigin(btVector3(-6., 0., 0.));
	
	btGeneric6DofConstraint* d6f2 = new btGeneric6DofConstraint(*body2, *body3, frameInA, frameInB, true);
	d6f2->setAngularLowerLimit(btVector3(0., SIMD_PI* .1, 0.));
    d6f2->setAngularUpperLimit(btVector3(0., SIMD_PI* .9, 0.));	
	_dynamicsWorld->addConstraint(d6f2);
	
	frameInA.setOrigin(btVector3(6., 0., 0.));
    frameInB.setOrigin(btVector3(-2., 0., 0.));
	
	btGeneric6DofConstraint* d6f3 = new btGeneric6DofConstraint(*body3, *body4, frameInA, frameInB, true);
	_dynamicsWorld->addConstraint(d6f3);
	
	body0->setDamping(12.9f, 12.9f);
	body1->setDamping(12.9f, 12.9f);
	body2->setDamping(12.9f, 12.9f);
	body3->setDamping(12.9f, 12.9f);
	body4->setDamping(12.9f, 12.9f);
	
	_drawer = new ShapeDrawer();

	m_activeBody = 0;
	
	m_testJoint = d6f;
	
	m_interactMode = TranslateBone;
}

void DynamicsSolver::killPhysics()
{
	//remove the rigidbodies from the dynamics world and delete them
	int i;
	for (i=_dynamicsWorld->getNumCollisionObjects()-1; i>=0 ;i--)
	{
		btCollisionObject* obj = _dynamicsWorld->getCollisionObjectArray()[i];
		btRigidBody* body = btRigidBody::upcast(obj);
		if (body && body->getMotionState())
		{

			while (body->getNumConstraintRefs())
			{
				btTypedConstraint* constraint = body->getConstraintRef(0);
				_dynamicsWorld->removeConstraint(constraint);
				delete constraint;
			}
			delete body->getMotionState();
			_dynamicsWorld->removeRigidBody(body);
		} 
		else
		{
			_dynamicsWorld->removeCollisionObject( obj );
		}
		delete obj;
	}
	
	for (int j=0;j<_collisionShapes.size();j++)
	{
		btCollisionShape* shape = _collisionShapes[j];
		delete shape;
	}

	delete _constraintSolver;

	delete _overlappingPairCache;

	delete _dispatcher;

	delete _collisionConfiguration;
}

void DynamicsSolver::renderWorld()
{
	const int	numObjects= _dynamicsWorld->getNumCollisionObjects();
	btVector3 wireColor(1,0,0);
	for(int i=0;i<numObjects;i++)
	{
		btCollisionObject*	colObj= _dynamicsWorld->getCollisionObjectArray()[i];
		_drawer->drawObject(colObj);
	}
	
	const int numConstraints = _dynamicsWorld->getNumConstraints();
	for(int i=0;i< numConstraints;i++) {
	    btTypedConstraint* constraint = _dynamicsWorld->getConstraint(i);
	    _drawer->drawConstraint(constraint);
	}
	
	if(!m_activeBody) return;
	
	if(m_interactMode == TranslateBone) {
	    _drawer->drawTranslateHandle(m_activeBody);
	}
	//else if(m_interactMode == RotateJoint) {
	    
	//}
}

void DynamicsSolver::simulate()
{
	btScalar dt = (btScalar)_clock.getTimeMicroseconds();
	_clock.reset();
	_dynamicsWorld->stepSimulation(dt / 1000000.f, 10);
}

btRigidBody* DynamicsSolver::localCreateRigidBody(float mass, const btTransform& startTransform,btCollisionShape* shape)
{
	btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0,0,0);
	if (isDynamic)
		shape->calculateLocalInertia(mass,localInertia);
	
	printf("inertial %f %f %f \n", localInertia.getX(), localInertia.getY(), localInertia.getZ());

	//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
	btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass,myMotionState,shape,localInertia);
	

	btRigidBody* body = new btRigidBody(cInfo);
	body->setContactProcessingThreshold(_defaultContactProcessingThreshold);

	return body;
}

char DynamicsSolver::selectByRayHit(const Vector3F & origin, const Vector3F & ray, Vector3F & hitP)
{
    m_activeBody = 0;
    btVector3 fromP(origin.x, origin.y, origin.z);
    btVector3 toP(origin.x + ray.x, origin.y + ray.y, origin.z + ray.z);
    btCollisionWorld::ClosestRayResultCallback rayCallback(fromP, toP);
    _dynamicsWorld->rayTest(fromP , toP, rayCallback);
    if(rayCallback.hasHit()) {
        btRigidBody * body = (btRigidBody *)btRigidBody::upcast(rayCallback.m_collisionObject);
        if(body) {
            body->setActivationState(DISABLE_DEACTIVATION);
            btVector3 pickPos = rayCallback.m_hitPointWorld;
            hitP.x = pickPos.getX();
            hitP.y = pickPos.getY();
            hitP.z = pickPos.getZ();
            m_activeBody = body;
            
            
            return 1;
        }
    }
    return 0;
}

void DynamicsSolver::addImpulse(const Vector3F & impulse)
{
    if(!m_activeBody) return;

    m_activeBody->setCollisionFlags(m_activeBody->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
    m_activeBody->setActivationState(DISABLE_DEACTIVATION);
    
    btTransform tm = m_activeBody->getWorldTransform();
    btVector3 t = tm.getOrigin() + btVector3(impulse.x/10.f, impulse.y/10.f, impulse.z/80.f);
    tm.setOrigin(t);
    
    m_activeBody->setWorldTransform(tm);
    m_activeBody->setLinearVelocity(btVector3(0.f, 0.f, 0.f));
    m_activeBody->setAngularVelocity(btVector3(0.f, 0.f, 0.f));
    m_activeBody->setCollisionFlags(m_activeBody->getCollisionFlags() & ~(btCollisionObject::CF_KINEMATIC_OBJECT));
    m_activeBody->forceActivationState(ACTIVE_TAG);
    
    //btVector3 impulseV(impulse.x, impulse.y, impulse.z);
    //m_activeBody->setActivationState(ACTIVE_TAG);
    //m_activeBody->applyForce(impulseV, btVector3(0,0,0));
}

void DynamicsSolver::addTorque(const Vector3F & torque)
{
    if(!m_activeBody) return;
    
    m_activeBody->setCollisionFlags(m_activeBody->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
    m_activeBody->setActivationState(DISABLE_DEACTIVATION);
    
    btTransform tm = m_activeBody->getWorldTransform();
    btQuaternion q = tm.getRotation();
    
    btQuaternion a;
    a.setRotation(btVector3(0.f, 0.f, 1.f), btScalar(1.57f));
    
    q = a;
    tm.setRotation(q);
    
    m_activeBody->setWorldTransform(tm);
    m_activeBody->setLinearVelocity(btVector3(0.f, 0.f, 0.f));
    m_activeBody->setAngularVelocity(btVector3(0.f, 0.f, 0.f));
    m_activeBody->setCollisionFlags(m_activeBody->getCollisionFlags() & ~(btCollisionObject::CF_KINEMATIC_OBJECT));
    //m_activeBody->forceActivationState(ACTIVE_TAG);
    
    //m_testJoint->getRotationalLimitMotor(0)->m_enableMotor = true;
    //m_testJoint->getRotationalLimitMotor(0)->m_targetVelocity = 4.0f;
    //m_testJoint->getRotationalLimitMotor(0)->m_maxMotorForce = 100.f;
}

void DynamicsSolver::removeTorque()
{
    m_testJoint->getRotationalLimitMotor(0)->m_enableMotor = false;
}

void DynamicsSolver::removeSelection()
{
    m_activeBody = 0;
}

char DynamicsSolver::hasActive() const
{
    return m_activeBody != 0;
}

void DynamicsSolver::toggleMassProp()
{
    if(!m_activeBody) return;
    
    if(m_activeBody->getInvMass() < 0.001f)
        m_activeBody->setMassProps(1.f, btVector3(0.666667, 0.666667, 0.666667));
    else
        m_activeBody->setMassProps(0.f, btVector3(0,0,0));
}

void DynamicsSolver::setInteractMode(InteractMode mode)
{
    m_interactMode = mode;
}

DynamicsSolver::InteractMode DynamicsSolver::getInteractMode() const
{
    return m_interactMode;
}

