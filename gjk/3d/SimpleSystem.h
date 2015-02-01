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
	
	virtual const Vector3F supportPoint(const Vector3F & v, const Matrix44F & space, Vector3F & localP) const
    {
        float maxdotv = -1e8;
        float dotv;
        
        Vector3F res;
        Vector3F worldP;
        
        for(int i=0; i < 4; i++) {
            worldP = space.transform(p[i]);
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
	    *angularMassM.m(0, 0) = 12.f / ( mass * (m_h * m_h + m_d * m_d));
	    *angularMassM.m(1, 1) = 12.f / ( mass * (m_w * m_w + m_d * m_d));
	    *angularMassM.m(2, 2) = 12.f / ( mass * (m_w * m_w + m_h * m_h));
	}
	
	virtual const float angularMotionDisc() const
    {
        return Vector3F(m_w, m_h, m_d).length();
    }
	
	virtual const Vector3F supportPoint(const Vector3F & v, const Matrix44F & space, Vector3F & localP) const
    {
        float maxdotv = -1e8;
        float dotv;
        
        Vector3F res;
        Vector3F worldP;
		
		Vector3F p[8];
		fillP(p);
        
        for(int i=0; i < 8; i++) {
            worldP = space.transform(p[i]);
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
	Quaternion orientation;
	Vector3F position;
	Vector3F linearVelocity;
	Vector3F angularVelocity;
	MassShape * shape;
};

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
	
	RigidBody * rb();
	RigidBody * ground();
private:
	void applyGravity();
	void applyImpulse();
	void applyVelocity();
	void continuousCollisionDetection(const RigidBody & A, const RigidBody & B);
private:
	RigidBody m_rb;
	RigidBody m_ground;
	GjkContactSolver m_gjk;
	ContinuousCollisionContext m_ccd;
	Vector3F * m_X;
	unsigned * m_indices;
	
	Vector3F * m_V;
	Vector3F * m_Vline;
	unsigned * m_vIndices;
	
	Vector3F * m_groundX;
	unsigned * m_groundIndices;
};
#endif        //  #ifndef SIMPLESYSTEM_H
