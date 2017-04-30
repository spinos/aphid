/*
 *  APointCloud.h
 *  aphid
 *
 *  Created by jian zhang on 4/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "Geometry.h"
class BaseBuffer;
class APointCloud : public Geometry {
public:
	APointCloud();
	virtual ~APointCloud();
	virtual const Type type() const;
	
	const unsigned numPoints() const;
	Vector3F * points() const;
	float * pointRadius() const;
	
	void create(unsigned n);
	void copyPointsFrom(Vector3F * src);
	
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	virtual bool intersectBox(const BoundingBox & box);
	virtual bool intersectTetrahedron(const Vector3F * tet);
	virtual bool intersectBox(unsigned icomponent, const BoundingBox & box);
	virtual bool intersectTetrahedron(unsigned icomponent, const Vector3F * tet);
protected:

private:
	BaseBuffer * m_points;
	BaseBuffer * m_radius;
	unsigned m_numPoints;
};