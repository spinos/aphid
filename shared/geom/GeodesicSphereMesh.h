/*
 *  GeodesicSphereMesh.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GEODESIC_SPHERE_MESH_H
#define APH_GEODESIC_SPHERE_MESH_H
#include <BaseMesh.h>
#include <geom/ATriangleMesh.h>

namespace aphid {

struct Float2;
    
class TriangleGeodesicSphere : public ATriangleMesh {
     
public:
    TriangleGeodesicSphere(int level = 5);
    virtual ~TriangleGeodesicSphere();
    
protected:

/// azimuth u [-pi,pi] v inclination [-0.5pi,0.5pi]    
	void computeSphericalCoord(Float2* dst) const;

private:
    void subdivide(int level, unsigned & currentVertex, unsigned & currentIndex, 
        Vector3F * p, unsigned * idx, 
        const Vector3F & a, const Vector3F & b, const Vector3F & c, const Vector3F & d);
    
};

class GeodesicSphereMesh : public BaseMesh {
public:
	GeodesicSphereMesh(unsigned level);
	virtual ~GeodesicSphereMesh();
	void setRadius(float r);
private:	
	void subdivide(unsigned level, unsigned & currentVertex, unsigned & currentIndex, Vector3F * p, unsigned * idx, Vector3F a, Vector3F b, Vector3F c, Vector3F d);
};

}
#endif