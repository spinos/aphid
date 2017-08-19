/*
 *  LoftMesh.h
 * 
 *  by connecting a number of profiles 
 *
 *  Created by jian zhang on 8/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_LOFT_MESH_H
#define APH_LOFT_MESH_H

#include "ATriangleMesh.h"

namespace aphid {

class LoftMeshBuilder {

	std::vector<Vector3F> m_points;
	std::vector<int> m_profileVertices;
	std::vector<int> m_profileBegins;
	std::vector<int> m_triangles;
	Vector3F m_defaultNml;
	Vector3F m_defaultCol;
	
public:
	LoftMeshBuilder();
	virtual ~LoftMeshBuilder();
	
	void addPoint(const Vector3F& v);
/// num and ind of profile vertices
	void addProfile(int nv, const int* vs);
/// triangulate by connecting a-th and b-th profile
/// a to b is counterclockwise
	void connectProfiles(int a, int b, bool isEven);
	
	int numProfileVertices() const;
	int numProfileBegins() const;
	
	int numTriangles() const;
	int numPoints() const;
	int numProfiles() const;
	
	int getProfileVertex(int i) const;
	int getProfileBegin(int i) const;
	void getPoint(Vector3F& dst, int i) const;
	void getTriangle(int* dst, int i) const;
	
	const Vector3F& defaultNormal() const;
	const Vector3F& defaultColor() const;
/// x-y plane	
	virtual void projectTexcoord(ATriangleMesh* msh,
					BoundingBox& bbx) const;
	
protected:

private:
/// c --- d
/// |  /  |
/// a --- b
	void addEvenTriangles(int a, int b, int c, int d);
/// c --- d
/// |  \  |
/// a --- b
	void addOddTriangles(int a, int b, int c, int d);
};

class LoftMesh : public ATriangleMesh {
	
	boost::scoped_array<int> m_profileVertices;
	boost::scoped_array<int> m_profileBegins;
	float m_width, m_height, m_depth;
	
public:
	LoftMesh();
	virtual ~LoftMesh();
	
	const float& width() const;
	const float& height() const;
	const float& depth() const;
	float widthHeightRatio() const;
	
protected:
	void createMesh(const LoftMeshBuilder& builder );
	
/// begin and end of i-th profile
	void getProfileRange(int& vbegin, int& vend,
			const int& i) const;
	const int& getProfileVertex(const int& i) const;
	
private:
};

}

#endif
