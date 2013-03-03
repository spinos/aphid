/*
 *  dynamicsSolver.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 

#include "shapeDrawer.h"
#include <Vector3F.h>
class btSoftRigidDynamicsWorld;
class btSoftRididCollisionAlgorithm;
class btSoftSoftCollisionAlgorithm;
class btDefaultCollisionConfiguration;

class DynamicsSolver {
public:
    enum InteractMode {
        ToggleLock,
        TranslateBone,
        RotateJoint
    };
    
	DynamicsSolver();
	virtual ~DynamicsSolver();
	
	void initPhysics();
	void killPhysics();
	void renderWorld();
	void simulate();
	char selectByRayHit(const Vector3F & origin, const Vector3F & ray, Vector3F & hitP);
	void addImpulse(const Vector3F & impulse);
	void addTorque(const Vector3F & torque);
	void removeTorque();
	void removeSelection();
	
	void setInteractMode(InteractMode mode);
	InteractMode getInteractMode() const;
	
	char hasActive() const;
	
	void toggleMassProp();
	
protected:
	btSoftRigidDynamicsWorld* m_dynamicsWorld;
	
	//class btBroadphaseInterface*	_overlappingPairCache;

	btCollisionDispatcher*	m_dispatcher;

	btConstraintSolver*	m_constraintSolver;

	btDefaultCollisionConfiguration* m_collisionConfiguration;
	
	btAlignedObjectArray<btCollisionShape*> m_collisionShapes;
	
	btScalar		_defaultContactProcessingThreshold;
	
	ShapeDrawer* _drawer;
		
	btClock _clock;

	btRigidBody* localCreateRigidBody(float mass, const btTransform& startTransform,btCollisionShape* shape);
	btRigidBody* m_activeBody;
	
	btGeneric6DofConstraint* m_testJoint;
	
	InteractMode m_interactMode;
	
	btBroadphaseInterface*	m_broadphase;

	
	btAlignedObjectArray<btSoftSoftCollisionAlgorithm*> m_SoftSoftCollisionAlgorithms;

	btAlignedObjectArray<btSoftRididCollisionAlgorithm*> m_SoftRigidCollisionAlgorithms;
private:
    
    void initRope();
    void relaxRope();

};