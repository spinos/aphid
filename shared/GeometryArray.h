#ifndef GEOMETRYARRAY_H
#define GEOMETRYARRAY_H

/*
 *  GeometryArray.h
 *  
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <Geometry.h>

class GeometryArray : public Geometry {
public:
	GeometryArray();
	virtual ~GeometryArray();
	
	void create(unsigned n);
	void setGeometry(Geometry * geo, unsigned i);
	void setNumGeometries(unsigned n);
	const unsigned numGeometries() const;
	Geometry * geometry(unsigned icomponent) const;
	void destroyGeometries();
	
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	virtual const Type type() const;
// overrid geometry
	virtual bool intersectRay(const Ray * r);
	virtual bool intersectRay(unsigned icomponent, const Ray * r);
protected:

private:
	Geometry ** m_geos;
	unsigned m_numGeometies;
};
#endif        //  #ifndef GEOMETRYARRAY_H
