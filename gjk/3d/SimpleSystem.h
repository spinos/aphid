#ifndef SIMPLESYSTEM_H
#define SIMPLESYSTEM_H

/*
 *  SimpleSystem.h
 *  proof
 *
 *  Created by jian zhang on 1/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "CollisionPair.h"

class SimpleSystem {
public:
	SimpleSystem();
	
	Vector3F * X() const;
	const unsigned numFaceVertices() const;
	unsigned * indices() const;
	
	Vector3F * groundX() const;
	const unsigned numGroundFaceVertices() const;
	unsigned * groundIndices() const;
	
	Vector3F * Vline() const;
	const unsigned numVlineVertices() const;
	unsigned * vlineIndices() const;
#ifdef DBG_DRAW	
	void setDrawer(KdTreeDrawer * d);
#endif	
	void progress();
	void drawWorld();
	
	RigidBody * rb();
	RigidBody * ground();
private:
	void applyGravity();
	void applyImpulse();
	void applyVelocity();
	void continuousCollisionDetection(const RigidBody & A, const RigidBody & B);
	void updateStates();
private:
	RigidBody m_rb;
	RigidBody m_ground;
	Vector3F * m_X;
	unsigned * m_indices;
	
	Vector3F * m_V;
	Vector3F * m_Vline;
	unsigned * m_vIndices;
	
	Vector3F * m_groundX;
	unsigned * m_groundIndices;
#ifdef DBG_DRAW	
	KdTreeDrawer * m_dbgDrawer;
#endif
};
#endif        //  #ifndef SIMPLESYSTEM_H
