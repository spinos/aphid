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
	
	template<typename T>
	void dumpComponent(T & acomp, const int & i) const;
	
	template<typename T>
	void dumpComponent(T & acomp, const int & i,
					const Matrix44F & tm) const;
	
protected:
	
private:
	
};

template<typename T>
void ATriangleMesh::dumpComponent(T & acomp, const int & i) const
{
	const Vector3F * p = points();
	const unsigned * v = triangleIndices(i);
	int a = v[0];
	int b = v[1];
	int c = v[2];
			
	acomp.setP(p[a], 0);
	acomp.setP(p[b], 1);
	acomp.setP(p[c], 2);
		
	const Float2 * uvs = triangleTexcoord(i);
	acomp.setUVs(uvs);
}

template<typename T>
void ATriangleMesh::dumpComponent(T & acomp, const int & i,
					const Matrix44F & tm) const
{
	const Vector3F * p = points();
	const Vector3F * nml = vertexNormals();
	const unsigned * v = triangleIndices(i);
	int a = v[0];
	int b = v[1];
	int c = v[2];
			
	acomp.setP(tm.transform(p[a]), 0);
	acomp.setP(tm.transform(p[b]), 1);
	acomp.setP(tm.transform(p[c]), 2);
	
	acomp.resetNC();
	Vector3F wn = tm.transformAsNormal(nml[a]);
	wn.normalize();
	acomp.setN(wn, 0);
	wn = tm.transformAsNormal(nml[b]);
	wn.normalize();
	acomp.setN(wn, 1);
	wn = tm.transformAsNormal(nml[c]);
	wn.normalize();
	acomp.setN(wn, 2);
		
	const Float2 * uvs = triangleTexcoord(i);
	acomp.setUVs(uvs);
}

}
#endif        //  #ifndef APH_GEOM_TRIANGLE_MESH_H
