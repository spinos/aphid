/*
 *  ConvexDomain.h
 *  
 *
 *  Created by jian zhang on 7/27/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "Boundary.h"
#include "ConvexShape.h"

namespace aphid {

template<typename T>
class ConvexDomain : public Domain {

	T * m_shape;
	
public:
	ConvexDomain(T * shape);

	virtual FunctionType functionType() const;
	virtual bool broadphaseIntersect(const BoundingBox & bx);
	virtual float distanceTo(const Vector3F & pref);
	virtual float beamIntersect(const Beam & b,
								const float & splatRadius);
	
	template<typename Ts>
	bool narrowphaseIntersect(const Ts * a, const float & shellThickness)
	{
		return m_shape->narrowphaseIntersect<Ts>(a, shellThickness);
	}
	
};

template<typename T>
ConvexDomain<T>::ConvexDomain(T * shape)
{ 
	m_shape = shape; 
	setBBox(shape->calculateBBox() );
}

template<typename T>
Domain::FunctionType ConvexDomain<T>::functionType() const
{ return T::FunctionTypeId; }

template<typename T>
bool ConvexDomain<T>::broadphaseIntersect(const BoundingBox & bx)
{ 
	if(!getBBox().intersect(bx) )
		return false;
		
	return m_shape->intersectBBox(bx);
}

template<typename T>
float ConvexDomain<T>::distanceTo(const Vector3F & pref)
{ return m_shape->distanceTo(pref); }

template<typename T>
float ConvexDomain<T>::beamIntersect(const Beam & b,
									const float & splatRadius)
{ return m_shape->beamIntersect(b); }

typedef ConvexDomain<cvx::Sphere> SphereDomain;
typedef ConvexDomain<cvx::Box> BoxDomain;

}