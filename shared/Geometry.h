/*
 *  Geometry.h
 *  kdtree
 *
 *  Created by jian zhang on 10/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <TypedEntity.h>
#include <AVerbose.h>
#include <Ray.h>
#include <BoundingBox.h>
#include <vector>
#include <GjkIntersection.h>
#include "BarycentricCoordinate.h"

namespace aphid {

class Geometry : public TypedEntity, public AVerbose {
public:
	struct ClosestToPointTestResult {
		BarycentricCoordinate _bar;
		Vector3F _toPoint;
		Vector3F _hitPoint;
		Vector3F _hitNormal;
		float _contributes[4];
		float _distance;
		unsigned _icomponent;
		bool _hasResult;
		bool _isInside;
		Geometry * _geom;
		
		ClosestToPointTestResult();
		
		void reset();
		void reset(const Vector3F & p, float initialDistance);
		bool closeTo(const BoundingBox & box);
		bool closeEnough();
	};
	
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