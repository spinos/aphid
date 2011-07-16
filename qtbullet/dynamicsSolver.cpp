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
	_dynamicsWorld->setGravity(btVector3(0,-10,0));

	btCollisionShape* groundShape = new btBoxShape(btVector3(75,1,75));
	_collisionShapes.push_back(groundShape);
	
	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(btVector3(0,-5,0));
	btRigidBody* body = localCreateRigidBody(0.f,tr,groundShape);


	_dynamicsWorld->addRigidBody(body);
	
	btCollisionShape* cubeShape = new btBoxShape(btVector3(1.f,1.f,1.f));
	_collisionShapes.push_back(cubeShape);
	
	for(int i=0; i<64; i++)
	{
		tr.setIdentity();
		tr.setOrigin(btVector3(0.05*i,6+2.0*i,0));
		btRigidBody* cube = localCreateRigidBody(1.f,tr,cubeShape);
		_dynamicsWorld->addRigidBody(cube);
	}
	
	
	_drawer = new ShapeDrawer();

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
	btScalar dt = (btScalar)_clock.getTimeMicroseconds();
	_clock.reset();
	_dynamicsWorld->stepSimulation(dt / 1000000.f, 10);
	btScalar	m[16];
	btMatrix3x3	rot;rot.setIdentity();
	const int	numObjects= _dynamicsWorld->getNumCollisionObjects();
	btVector3 wireColor(1,0,0);
	for(int i=0;i<numObjects;i++)
	{
		btCollisionObject*	colObj= _dynamicsWorld->getCollisionObjectArray()[i];
		btRigidBody*		body=btRigidBody::upcast(colObj);
		if(body&&body->getMotionState())
		{
			btDefaultMotionState* myMotionState = (btDefaultMotionState*)body->getMotionState();
			myMotionState->m_graphicsWorldTrans.getOpenGLMatrix(m);
			rot=myMotionState->m_graphicsWorldTrans.getBasis();
		}
		
		_drawer->draw(m,colObj->getCollisionShape());
	}
}

btRigidBody* DynamicsSolver::localCreateRigidBody(float mass, const btTransform& startTransform,btCollisionShape* shape)
{
	btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0,0,0);
	if (isDynamic)
		shape->calculateLocalInertia(mass,localInertia);

	//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
	btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass,myMotionState,shape,localInertia);
	

	btRigidBody* body = new btRigidBody(cInfo);
	body->setContactProcessingThreshold(_defaultContactProcessingThreshold);

	return body;
}