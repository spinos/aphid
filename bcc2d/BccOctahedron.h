/*
 *  BccOctahedron.h
 *  
 *
 *  Created by jian zhang on 4/28/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
class BccOctahedron {
public:
	BccOctahedron();
	virtual ~BccOctahedron();
	void create(const Vector3F & center, float size);
	float movePoleCost(int * v, BccOctahedron & another);
	float moveEdgeCost(int * v, BccOctahedron & another);
	
	Vector3F * p();
	int * vertexTag();
	unsigned * vertexIndex();
	
	void getEdge(Vector3F & a, Vector3F & b, int idx);
	void getEdgeVertices(int & a, int & b, int idx);
	
	void createTetrahedron(std::vector<Vector3F > & points, std::vector<unsigned > & indices);
	
	static void movePoles(BccOctahedron & octa1, int va, BccOctahedron & octa2, int vb, std::vector<Vector3F > & points);
	static void moveEdges(BccOctahedron & octa1, int ea, BccOctahedron & octa2, int eb, std::vector<Vector3F > & points);
private:
	
private:
	Vector3F m_p[6];
	int m_tag[6];
	unsigned m_meshVerticesInd[6];
};
