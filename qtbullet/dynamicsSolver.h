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
#include <Vector3F.h>

class DynamicsSolver {
public:
    enum InteractMode {
        ToggleLock,
        TranslateBone,
        RotateJoint
    };
	DynamicsSolver() {}
	virtual ~DynamicsSolver() {}
	
	void initPhysics();
	void killPhysics();
	void renderWorld();
	void simulate();
	char selectByRayHit(const Vector3F & origin, const Vector3F & ray, Vector3F & hitP);
	void addImpulse(const Vector3F & impulse);
	void addTorque(const Vector3F & torque);
	void removeTorque();
	
	void setInteractMode(InteractMode mode);
	InteractMode getInteractMode() const;
	
	char hasActive() const;
	
	void toggleMassProp();
	
	InteractMode m_interactMode;
	
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
	btRigidBody* m_activeBody;
	
	btGeneric6DofConstraint* m_testJoint;
};