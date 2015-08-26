/*
 *  BlockBccMeshBuilder.h
 *  bcc
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <TetrahedronMeshBuilder.h>
#include <AOrientedBox.h>
#include <deque>
class CartesianGrid;
class GeometryArray;
class BlockBccMeshBuilder : public TetrahedronMeshBuilder {
public:
	BlockBccMeshBuilder();
	virtual ~BlockBccMeshBuilder();
	
	virtual void build(GeometryArray * geos,
				unsigned & ntet, unsigned & nvert, unsigned & nstripes);
	 
	virtual void addAnchors(ATetrahedronMesh * mesh, unsigned n, KdIntersection * anchorMesh);
	
	static unsigned MinimumUGrid;
	static unsigned MinimumVGrid;
	static unsigned MinimumWGrid;
	static unsigned MaximumUGrid;
	static unsigned MaximumVGrid;
	static unsigned MaximumWGrid;
protected:
	void build(AOrientedBox * ob, 
				int gx, int gy, int gz);
	
    void addTetrahedron(Vector3F * v, unsigned * ind);
private:
	void addNorth(const Vector3F & center, float size, float hsize);
	void addEast(const Vector3F & center, float size, float hsize, int i, int n);
	void addDepth(const Vector3F & center, float size, float hsize, int i, int n);
private:
    CartesianGrid * m_verticesPool;
};