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
#include "ConvexDomain.h"
#include <NTreeDomain.h>

namespace aphid {

class BDistanceFunction {

	std::vector<Domain *> m_domains;

public:
	BDistanceFunction();
	virtual ~BDistanceFunction();

	void addSphere(const Vector3F & p, const float & r);
	void addBox(const Vector3F & lo, const Vector3F & hi);
	
	template<typename T, typename Tn>
	void addTree(KdNTree<T, Tn > * tree)
	{ m_domains.push_back(new NTreeDomain<T, Tn>(tree) ); }
	
/// limit distance test of all domains
	void setDomainDistanceRange(const float & x);
	float calculateDistance(const Vector3F & p);
/// closest intersect v a->b, alpha of cross
	float calculateIntersection(const Vector3F & a,
								const Vector3F & b);
	
	template<typename Ts>
	bool intersect(const Ts * a, const float & shellThickness) const
	{
		BoundingBox ab = a->calculateBBox();
		ab.expand(shellThickness);
		
		std::vector<Domain *>::const_iterator it = m_domains.begin();
		for(;it!=m_domains.end();++it) {
			
			Domain * d = *it;
			if(d->broadphaseIntersect(ab) )
				return true;
			
		}
		return false;
	}
	
protected:
	
	
private:
	void internalClear();
	
};

}