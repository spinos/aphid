#ifndef APH_GEOM_TRIANGLE_MESH_H
#define APH_GEOM_TRIANGLE_MESH_H

/*
 *  ATriangleMesh.h
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <geom/AGenericMesh.h>
#include <boost/scoped_array.hpp>

namespace aphid {

class ATriangleMesh : public AGenericMesh {
	
	gjk::TriangleSet m_componentTriangle;
/// face varying
	boost::scoped_array<Float2> m_texcoord;
	
public:
	ATriangleMesh();
	virtual ~ATriangleMesh();
	
	virtual const Type type() const;
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	virtual void closestToPoint(unsigned icomponent, ClosestToPointTestResult * result);
	virtual bool intersectTetrahedron(unsigned icomponent, const Vector3F * tet);
	virtual bool intersectRay(unsigned icomponent, const Ray * r,
					Vector3F & hitP, Vector3F & hitN, float & hitDistance);
	virtual bool intersectSphere(unsigned icomponent, const gjk::Sphere & B);
	virtual bool intersectBox(unsigned icomponent, const BoundingBox & box);
	const unsigned numTriangles() const;
	void setTriangleTexcoord(const int & idx, const Float2 * uvs);
	const Float2 * triangleTexcoord(const int & idx) const;
	
	void create(unsigned np, unsigned nt);
	unsigned * triangleIndices(unsigned idx) const;
	const Vector3F triangleCenter(unsigned idx) const;
	const Vector3F triangleNormal(unsigned idx) const;
	virtual std::string verbosestr() const;
	
	void reverseTriangleNormals();
    void calculateVertexNormals();
	
protected:
	
private:
	
};

}
#endif        //  #ifndef APH_GEOM_TRIANGLE_MESH_H
