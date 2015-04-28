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
	void create(const Vector3F & center, const Vector3F & dir);
	float movePoleCost(int * v, BccOctahedron & another);
	float moveEdgeCost(int * v, BccOctahedron & another);
	void closestPoleTo(int & va, int & vb,  BccOctahedron & another);
	void closestEdge89To(int & ea, int & eb,  BccOctahedron & another, int pole);
	
	Vector3F * p();
	int * vertexTag();
	unsigned * vertexIndex();
	const int axis() const;
	const float size() const;
	
	void getEdge(Vector3F & a, Vector3F & b, int idx);
	void getEdgeVertices(int & a, int & b, int idx);
	
	void createTetrahedron(std::vector<Vector3F > & points, std::vector<unsigned > & indices);
	
	static void movePoles(BccOctahedron & octa1, int va, BccOctahedron & octa2, int vb, std::vector<Vector3F > & points);
	static void moveEdges(BccOctahedron & octa1, int ea, BccOctahedron & octa2, int eb, std::vector<Vector3F > & points);
	static void add8GapTetrahedron(BccOctahedron & octa1, int va, 
	                               BccOctahedron & octa2, int vb,
	                               std::vector<unsigned > & indices);
	static void add2GapTetrahedron(BccOctahedron & octa1, int ea, 
	                               BccOctahedron & octa2, int eb,
	                               std::vector<unsigned > & indices);
	static void connectDifferentAxis(BccOctahedron & octa1,
										BccOctahedron & octa2,
										std::vector<Vector3F > & points);
private:
	
private:
	Vector3F m_p[6];
	int m_tag[6];
	unsigned m_meshVerticesInd[6];
	int m_axis;
	float m_size;
};
