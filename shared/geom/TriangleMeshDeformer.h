/*
 *  TriangleMeshDeformer.h
 *  
 *  deform a billboard
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TRIANGLE_MESH_DEFORMER_H
#define APH_TRIANGLE_MESH_DEFORMER_H

#include <boost/scoped_array.hpp>

namespace aphid {

class Vector3F;
class ATriangleMesh;

class TriangleMeshDeformer {

	boost::scoped_array<Vector3F > m_points;
	boost::scoped_array<Vector3F > m_normals;
/// num point to deform
	int m_np;
	
public:
    TriangleMeshDeformer();
	virtual ~TriangleMeshDeformer();
	
	virtual void deform(const ATriangleMesh * mesh);
	
	const Vector3F * deformedPoints() const;
	const Vector3F * deformedNormals() const;
	const int& numPoints() const;
	
	void updateGeom(ATriangleMesh* outMesh,
				const ATriangleMesh* inMesh);
	
protected:
	Vector3F* points();
	Vector3F* normals();

	float getRowMean(int rowBegin, int nv, int& nvRow, float& rowBase ) const;
	void setOriginalMesh(const ATriangleMesh * mesh);
	void calculateNormal(const ATriangleMesh * mesh);

	static int GetNumRows(const ATriangleMesh * mesh);
	
private:
	static int GetRowNv(float& rowBase, int rowBegin, int nv, const Vector3F* ps );
	
};

}
#endif
