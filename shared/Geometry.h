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
#include <BoundingBox.h>
class Geometry : public TypedEntity, public AVerbose {
public:
	struct ClosestToPointTestResult {
		Vector3F _toPoint;
		Vector3F _hitPoint;
		Vector3F _hitNormal;
		float _tricoord[3];
		float _distance;
		unsigned _icomponent;
		bool _hasResult;
		void reset(const Vector3F & p, float d) {
			_toPoint = p;
			_distance = d;
			_hasResult = false;
		}
		bool closeTo(const BoundingBox & box) {
			return box.distanceTo(_toPoint) < _distance;
		}
	};
	
	Geometry();
	virtual ~Geometry();
	
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	virtual bool intersectBox(const BoundingBox & box);
	virtual bool intersectTetrahedron(const Vector3F * tet);
	virtual bool intersectRay(const Ray * r);
	virtual bool intersectBox(unsigned icomponent, const BoundingBox & box);
	virtual bool intersectTetrahedron(unsigned icomponent, const Vector3F * tet);
	virtual bool intersectRay(unsigned icomponent, const Ray * r);
	virtual void closestToPoint(ClosestToPointTestResult * result);
	virtual void closestToPoint(unsigned icomponent, ClosestToPointTestResult * result);
protected:
    
private:
};