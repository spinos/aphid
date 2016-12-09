/*
 *  Geometry.h
 *  kdtree
 *
 *  Created by jian zhang on 10/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <foundation/TypedEntity.h>
#include <foundation/AVerbose.h>
#include <foundation/NamedEntity.h>
#include <GjkIntersection.h>
#include <Ray.h>

namespace aphid {

class ClosestToPointTestResult;

class Geometry : public TypedEntity, public NamedEntity, public AVerbose {
public:	
	Geometry();
	virtual ~Geometry();
	
	const Vector3F boundingCenter() const;
	
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	virtual bool intersectBox(const BoundingBox & box);
	virtual bool intersectTetrahedron(const Vector3F * tet);
	virtual bool intersectRay(const Ray * r);
	virtual bool intersectBox(unsigned icomponent, const BoundingBox & box);
	virtual bool intersectTetrahedron(unsigned icomponent, const Vector3F * tet);
	virtual bool intersectRay(unsigned icomponent, const Ray * r,
					Vector3F & hitP, Vector3F & hitN, float & hitDistance);
	virtual bool intersectSphere(unsigned icomponent, const gjk::Sphere & B);
	virtual void closestToPoint(ClosestToPointTestResult * result);
	virtual void closestToPointElms(const std::vector<unsigned > & elements, ClosestToPointTestResult * result);
	virtual void closestToPoint(unsigned icomponent, ClosestToPointTestResult * result);
protected:
    
private:
};

}