/*
 *  CollisionPair.h
 *  proof
 *
 *  Created by jian zhang on 2/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef COLLISIONPAIR_H
#define COLLISIONPAIR_H
#include "GjkContactSolver.h"

class MassShape : public PointSet
{
public:
    MassShape() {}
    virtual ~MassShape() {}
    virtual void setMass(float mass) {}
    Vector3F linearMassM;
	Matrix33F angularMassM;
};

class TetrahedronShape : public MassShape {
public:
	TetrahedronShape() {}
	
	Aabb computeAabb() const {
	    Aabb box;
		resetAabb(box);
        for(int i=0; i < 4; i++) 
			expandAabb(box, p[i]);
		return box;
	}
	
	virtual void setMass(float mass)
	{
	    linearMassM.x = linearMassM.y = linearMassM.z = 1.f / mass;
	    angularMassM.setIdentity();
	    
	    Aabb box = computeAabb();
	    Vector3F sz = box.high - box.low;
	    sz *= 0.5f;
		
	    // cuboid approximation
	    *angularMassM.m(0, 0) = 12.f / ( mass * (sz.y * sz.y + sz.z * sz.z));
	    *angularMassM.m(1, 1) = 12.f / ( mass * (sz.x * sz.x + sz.z * sz.z));
	    *angularMassM.m(2, 2) = 12.f / ( mass * (sz.x * sz.x + sz.y * sz.y));
	}
	
	virtual const float angularMotionDisc() const
    {
		Aabb box = computeAabb();

        const Vector3F center = box.low * 0.5f + box.high * 0.5f;
        const Vector3F d = box.high - box.low;
        return center.length() + d.length() * 0.5f;
    }
	
	virtual const Vector3F supportPoint(const Vector3F & v, const Matrix44F & space, Vector3F & localP, const float & margin) const
    {
        float maxdotv = -1e8;
        float dotv;
        
        Vector3F res;
        Vector3F worldP;
        const Vector3F mar = v.normal() * margin;
		
        for(int i=0; i < 4; i++) {
            worldP = space.transform(p[i]) + mar;
            dotv = worldP.dot(v);
            if(dotv > maxdotv) {
                maxdotv = dotv;
                res = worldP;
                localP = p[i];
            }
        }
        
        return res;
    }
	
	Vector3F p[4];
};

class CuboidShape : public MassShape {
public:
	CuboidShape(float w, float h, float d)
	{
		m_w = w;
		m_h = h;
		m_d = d;
	}
	
	virtual void setMass(float mass)
	{
	    linearMassM.x = linearMassM.y = linearMassM.z = 1.f / mass;
	    angularMassM.setIdentity();
	    *angularMassM.m(0, 0) = 3.f / ( mass * (m_h * m_h + m_d * m_d));
	    *angularMassM.m(1, 1) = 3.f / ( mass * (m_w * m_w + m_d * m_d));
	    *angularMassM.m(2, 2) = 3.f / ( mass * (m_w * m_w + m_h * m_h));
	}
	
	virtual const float angularMotionDisc() const
    {
        return Vector3F(m_w, m_h, m_d).length();
    }
	
	virtual const Vector3F supportPoint(const Vector3F & v, const Matrix44F & space, Vector3F & localP, const float & margin) const
    {
        float maxdotv = -1e8;
        float dotv;
        
        Vector3F res;
        Vector3F worldP;
		const Vector3F mar = v.normal() * margin;
		
		Vector3F p[8];
		fillP(p);
		
        for(int i=0; i < 8; i++) {
            worldP = space.transform(p[i]) + mar;
            dotv = worldP.dot(v);
            if(dotv > maxdotv) {
                maxdotv = dotv;
                res = worldP;
                localP = p[i];
            }
        }
        
        return res;
    }
	
	void fillP(Vector3F * p) const
	{
		p[0].set(-m_w, -m_h, -m_d);
        p[1].set( m_w, -m_h, -m_d);
        p[2].set(-m_w,  m_h, -m_d);
        p[3].set( m_w,  m_h, -m_d);
        p[4].set(-m_w, -m_h,  m_d);
        p[5].set( m_w, -m_h,  m_d);
        p[6].set(-m_w,  m_h,  m_d);
        p[7].set( m_w,  m_h,  m_d);
	}

	float m_w, m_h, m_d; 
};

struct RigidBody {
	Matrix33F inertiaTensor;
	Quaternion orientation;
	Vector3F position;
	Vector3F linearVelocity;
	Vector3F angularVelocity;
	Vector3F projectedLinearVelocity;
	Vector3F projectedAngularVelocity;
	float TOI, Crestitution;
#ifdef DBG_DRAW
	Vector3F r, J;
	float Jsize;
#endif
	MassShape * shape;
	void integrateP(const float & h) {
		position = position.progress(linearVelocity, h * TOI * .99f);
		orientation = orientation.progress(angularVelocity, h * TOI * .99f);
		if(TOI < 1.f) {
			position = position.progress(projectedLinearVelocity, h * (1.f - TOI));
			orientation = orientation.progress(projectedAngularVelocity, h * (1.f - TOI));
		}
	}
	void updateState() {
		if(TOI < 1.f) {
			linearVelocity = projectedLinearVelocity;
			angularVelocity = projectedAngularVelocity;
		}
		
		computeInertiaTensor();
	}
	void computeInertiaTensor()
	{
		Matrix33F A; A.set(orientation);
		Matrix33F At = A; At.transpose();
		Matrix33F AAt = A * At;
		inertiaTensor = A * shape->angularMassM *At;
		// std::cout<<" AIinvAt "<<inertiaTensor.str();
	}
};

class CollisionPair {
public:
	CollisionPair(RigidBody * a, RigidBody * b);
	void continuousCollisionDetection(const float & h);
	void progressToImpactPostion(const float & h);
	void progressOnImpactPostion(const float & h);
	void detectAtImpactPosition(const float & h);
	const char hasContact() const;
	const float TOI() const;
	const Vector3F relativeVelocity() const;
	const Vector3F velocityAtContactA() const;
	const Vector3F velocityAtContactB() const;
	const Vector3F angularMotionAtContactB() const;
	void computeLinearImpulse(float & MinvJa, float & MinvJb, Vector3F & N);
	void computeAngularImpulse(Vector3F & IinvJa, float & MinvJa, Vector3F & IinvJb, float & MinvJb);
	void getTransformA(Matrix44F & t) const;
	void getTransformB(Matrix44F & t) const;
	
#ifdef DBG_DRAW	
	void setDrawer(KdTreeDrawer * d);
#endif
private:
	RigidBody * m_A;
	RigidBody * m_B;
	GjkContactSolver m_gjk;
	ContinuousCollisionContext m_ccd;
};

#endif
