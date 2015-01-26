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
#include "Gjk.h"

struct RigidBody {
	Quaternion orientation;
	Vector3F position;
	Vector3F linearVelocity;
	Vector3F angularVelocity;
	PointSet * shape;
};

class TetrahedronShape : public PointSet {
public:
	TetrahedronShape() {}
	virtual const Vector3F supportPoint(const Vector3F & v) const
    {
		float maxD = -1e8;
		float d;
		Vector3F r;
		for(int i=0; i < 4; i++) {
			d = p[i].dot(v);
			if(d > maxD) {
				maxD = d;
				r = p[i];
			}
		}
		return r;
	}
	Vector3F p[4];
};

class CuboidShape : public PointSet {
public:
	CuboidShape(float w, float h, float d)
	{
		m_w = w;
		m_h = h;
		m_d = d;
	}
	
	virtual const Vector3F supportPoint(const Vector3F & v) const
    {
		Vector3F p(-m_w, -m_h, -m_d);
		
        Vector3F res = p;
        float mdotv = p.dot(v);
		
		p.set(m_w, -m_h, -m_d);
        float dotv = p.dot(v);
        if(dotv > mdotv) {
            mdotv = dotv;
            res = p;
        }
        
		p.set(-m_w, m_h, -m_d);
        dotv = p.dot(v);
        if(dotv > mdotv) {
            mdotv = dotv;
            res = p;
        }
		
		p.set(m_w, m_h, -m_d);
        dotv = p.dot(v);
        if(dotv > mdotv) {
            mdotv = dotv;
            res = p;
        }
		
		p.set(-m_w, -m_h, m_d);
        dotv = p.dot(v);
        if(dotv > mdotv) {
            mdotv = dotv;
            res = p;
        }
		
		p.set(m_w, -m_h, m_d);
        dotv = p.dot(v);
        if(dotv > mdotv) {
            mdotv = dotv;
            res = p;
        }
        
		p.set(-m_w, m_h, m_d);
        dotv = p.dot(v);
        if(dotv > mdotv) {
            mdotv = dotv;
            res = p;
        }
		
		p.set(m_w, m_h, m_d);
        dotv = p.dot(v);
        if(dotv > mdotv) {
            mdotv = dotv;
            res = p;
        }
        return res;
    }

	float m_w, m_h, m_d; 
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
	
	void progress();
	
	RigidBody * rb();
	RigidBody * ground();
private:
	void applyGravity();
	void applyVelocity();
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
};
#endif        //  #ifndef SIMPLESYSTEM_H
