/*
 *  Boundary.h
 *  aphid
 *
 *  Created by jian zhang on 4/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BoundingBox.h>
namespace aphid {

class Boundary {
	BoundingBox m_bbox;
	
public:
	Boundary();
	
	void setBBox(const BoundingBox &bbox);
	const BoundingBox & getBBox() const;
	
protected:
	void updateBBox(const BoundingBox & box);
	BoundingBox * bbox();
private:
	
};

/*
 *  boundary with type and functions
 *
 */
class Domain : public Boundary {

public:
	Domain();
	
	enum FunctionType {
		fnUnknown = 0,
		fnSphere = 1,
		fnBox = 2,
		fnKdTree = 3,
		fnTetrahedron = 4
	};
	
	virtual FunctionType functionType() const;
	virtual bool broadphaseIntersect(const BoundingBox & b);
	virtual bool narrowphaseIntersect(Domain * b);
	virtual float distanceTo(const Vector3F & pref);
	
};

}