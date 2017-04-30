/*
 *  dynamicsSolver.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#pragma once
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
	void setEnableDrawConstraint(bool x);
	void setNumSubSteps(int x);
	void setSimulateFrequency(float x);
	void setSimulateScale(const float & x);
	const float simulateScale() const;
	const bool isPhysicsEnabled() const;
	const bool isWorldInitialized() const;
	const int numCollisionObjects() const;
	
	btBoxShape* createBoxShape(const float & x, const float & y, const float & z);
	btCylinderShape* createCylinderShape(const float & x, const float & y, const float & z);
	btSphereShape* createSphereShape(const float & r);
	void addCollisionShape(btCollisionShape* shape);
	
	btRigidBody* getRigidBody(const int & i) const;
	const Matrix44F restoreTM(const btTransform & tm) const;
	
	void addGroundPlane(const float & groundSize, const float & groundLevel);
	
	ShapeDrawer* getDrawer();
	const float deltaTime() const;
	
	btRigidBody* createRigidBody(btCollisionShape* shape, const Matrix44F & transform, const float & mass, bool multiplyTM = false);
	btGeneric6DofConstraint* constrainByHinge(btRigidBody& rbA, btRigidBody& rbB, const Matrix44F & rbAFrame, const Matrix44F & rbBFrame, bool disableCollisionsBetweenLinkedBodies=false);
	btGeneric6DofConstraint* constrainBy6Dof(btRigidBody& rbA, btRigidBody& rbB, const Matrix44F & rbAFrame, const Matrix44F & rbBFrame, bool disableCollisionsBetweenLinkedBodies=false);
	btGeneric6DofSpringConstraint* constrainBySpring(btRigidBody& rbA, btRigidBody& rbB, const Matrix44F & rbAFrame, const Matrix44F & rbBFrame, bool disableCollisionsBetweenLinkedBodies=false); 
	
protected:
    btCollisionObject * getCollisionObject(const int & i) const;
	
	virtual void clientBuildPhysics();
	
private:
    btRigidBody* createRigidBody(btCollisionShape* shape, const btTransform & transform, const float & mass);
	btGeneric6DofConstraint* constrainByHinge(btRigidBody& rbA, btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool disableCollisionsBetweenLinkedBodies=false);
	btGeneric6DofConstraint* constrainBy6Dof(btRigidBody& rbA, btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool disableCollisionsBetweenLinkedBodies=false);
	btGeneric6DofSpringConstraint* constrainBySpring(btRigidBody& rbA, btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool disableCollisionsBetweenLinkedBodies=false); 
	
	btRigidBody* internalCreateRigidBody(float mass, const btTransform& startTransform, btCollisionShape* shape);
	
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
	
	Matrix44F m_simulateSpace;
	int m_numSubSteps;
	float m_simulateFrequency;
	float m_dt;
	float m_simulateScale;
	bool m_enablePhysics;
	bool m_isWorldInitialized;
	bool m_enableDrawConstraint;
};