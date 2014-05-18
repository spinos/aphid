/*
 *  dynamicsSolver.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 

#include "shapeDrawer.h"
#include <AllMath.h>
class btSoftRigidDynamicsWorld;
class btSoftRididCollisionAlgorithm;
class btSoftSoftCollisionAlgorithm;
class btDefaultCollisionConfiguration;

class DynamicsSolver {
public:    
	DynamicsSolver();
	virtual ~DynamicsSolver();
	
	void initPhysics();
	void killPhysics();
	void renderWorld();
	void simulate();
	
protected:
	virtual void clientBuildPhysics();
	btRigidBody* createRigidBody(float mass, const btTransform& startTransform, btCollisionShape* shape);
	
	void addGroundPlane(const float & groundSize, const float & groundLevel);
	btBoxShape* createBoxShape(const float & x, const float & y, const float & z);
	btRigidBody* createRigitBox(btCollisionShape* shape, const btTransform & transform, const float & mass);
	btHingeConstraint* constrainByHinge(btRigidBody& rbA, btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool disableCollisionsBetweenLinkedBodies=false); 
private:
	btDynamicsWorld* m_dynamicsWorld;
	
	btBroadphaseInterface*	m_overlappingPairCache;

	btCollisionDispatcher*	m_dispatcher;

	btConstraintSolver*	m_constraintSolver;

	btDefaultCollisionConfiguration* m_collisionConfiguration;
	
	btScalar		_defaultContactProcessingThreshold;
	
	ShapeDrawer* _drawer;
		
	btClock _clock;
	
	btAlignedObjectArray<btCollisionShape*> m_collisionShapes;

	btAlignedObjectArray<btSoftSoftCollisionAlgorithm*> m_SoftSoftCollisionAlgorithms;

	btAlignedObjectArray<btSoftRididCollisionAlgorithm*> m_SoftRigidCollisionAlgorithms;
};