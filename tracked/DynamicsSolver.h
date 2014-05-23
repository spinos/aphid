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
	void simulate(const float & dt, const int & numSubsteps, const float & frequency);
	
	void setEnablePhysics(bool x);
	void setNumSubSteps(int x);
	const bool isWorldInitialized() const;
	const int numCollisionObjects() const;
	
	btBoxShape* createBoxShape(const float & x, const float & y, const float & z);
	btCylinderShape* createCylinderShape(const float & x, const float & y, const float & z);
	btSphereShape* createSphereShape(const float & r);
	
	btRigidBody* createRigitBody(btCollisionShape* shape, const btTransform & transform, const float & mass);
	
	btRigidBody* getRigidBody(const int & i) const;
	
protected:
    btCollisionObject * getCollisionObject(const int & i) const;
	
	virtual void clientBuildPhysics();
	btRigidBody* createRigidBody(float mass, const btTransform& startTransform, btCollisionShape* shape);
	
	void addGroundPlane(const float & groundSize, const float & groundLevel);
	
	btGeneric6DofConstraint* constrainByHinge(btRigidBody& rbA, btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool disableCollisionsBetweenLinkedBodies=false);
	btGeneric6DofSpringConstraint* constrainBySpring(btRigidBody& rbA, btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool disableCollisionsBetweenLinkedBodies=false); 
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
	
	bool m_enablePhysics;
	bool m_isWorldInitialized;
	int m_numSubSteps;
};