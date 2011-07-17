/*
 *  dynamicsSolver.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "btBulletDynamicsCommon.h"
#include "btBulletCollisionCommon.h"

#include "shapeDrawer.h"


class DynamicsSolver {
public:
	DynamicsSolver() {}
	virtual ~DynamicsSolver() {}
	
	void initPhysics();
	void killPhysics();
	void renderWorld();
	void simulate();
	
	btDynamicsWorld*		getDynamicsWorld()
	{
		return _dynamicsWorld;
	}
protected:
	btDynamicsWorld* _dynamicsWorld;
	class btBroadphaseInterface*	_overlappingPairCache;

	class btCollisionDispatcher*	_dispatcher;

	class btConstraintSolver*	_constraintSolver;

	class btDefaultCollisionConfiguration* _collisionConfiguration;
	
	btAlignedObjectArray<btCollisionShape*> _collisionShapes;
	
	btScalar		_defaultContactProcessingThreshold;
	
	ShapeDrawer* _drawer;
		
	btClock _clock;

	btRigidBody* localCreateRigidBody(float mass, const btTransform& startTransform,btCollisionShape* shape);

};