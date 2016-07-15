/*
 *  BDistanceFunction.h
 *  
 *	distance to a number of convex shapes
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "ConvexShape.h"

namespace aphid {

class BDistanceFunction {

	std::vector<cvx::Shape *> m_shapes;

public:
	BDistanceFunction();
	virtual ~BDistanceFunction();

	void addSphere(const Vector3F & p, const float & r);
	void addBox(const Vector3F & lo, const Vector3F & hi);
	void addFunction(cvx::Shape * d);
	float calculateDistance(const Vector3F & p) const;
	
	template<typename Ts>
	bool intersect(const Ts * a) const
	{
		std::vector<cvx::Shape *>::const_iterator it = m_shapes.begin();
		for(;it!=m_shapes.end();++it) {
			
			const cvx::Shape * b = *it;
			const cvx::ShapeType bt = b->shapeType();
			
			if(bt == aphid::cvx::TSphere) {
				if(sphereIntersect<Ts>(a, b) )
					return true;
			}
			else if(bt == aphid::cvx::TCube) {
				if(cubeIntersect<Ts>(a, b) )
					return true;
			}
			else if(bt == aphid::cvx::TBox) {
				if(boxIntersect<Ts>(a, b) )
					return true;
			}	
			
		}
		return false;
	}
	
protected:
	template<typename Ts>
	bool sphereIntersect(const Ts * a, const aphid::cvx::Shape * b) const 
	{ 
		const aphid::cvx::Sphere * sp = (const aphid::cvx::Sphere *)b;	
		return sp->intersect<Ts>(a); 
	}
	
	template<typename Ts>
	bool cubeIntersect(const Ts * a, const aphid::cvx::Shape * b) const 
	{ 
		const aphid::cvx::Cube * sp = (const aphid::cvx::Cube *)b;	
		return sp->intersect<Ts>(a); 
	}
	
	template<typename Ts>
	bool boxIntersect(const Ts * a, const aphid::cvx::Shape * b) const 
	{ 
		const aphid::cvx::Box * sp = (const aphid::cvx::Box *)b;	
		return sp->intersect<Ts>(a); 
	}
	
private:
	void internalClear();
	
};

}