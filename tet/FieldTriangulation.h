/*
 *  FieldTriangulation.h
 *  foo
 *
 *  red-blue refine each tetrahedron
 *  Created by jian zhang on 7/25/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "AdaptiveBccField.h"
#include <ttg/RedBlueRefine.h>

namespace aphid {
namespace ttg {

class FieldTriangulation : public AdaptiveBccField {
	
	sdb::Coord3 * m_triInds;
	Vector3F * m_cutPosBuf;
	Vector3F * m_vertexX;
	Vector3F * m_vertexN;
	int m_maxCutPosBuf, m_numAddedVert;
	int m_numFrontTriangleVertices, m_numFrontTris;
	
public:
	FieldTriangulation();
	virtual ~FieldTriangulation();
	
	void triangulateFront();
	
	const int & numFrontTriangles() const;
	void getTriangleShape(cvx::Triangle & t, const int & i) const;	
	
	const int & numAddedVertices() const;
	const Vector3F & addedVertex(const int & i) const;

	const Vector3F * triangleVertexP() const;
	const Vector3F * triangleVertexN() const;
	const int & numTriangleVertices() const;
	const int * triangleIndices() const;
	
protected:
	
private:
	struct ICutEdge {
		sdb::Coord2 key;
		int ind;
	};
	
	enum CutIndMask {
		MEncode = 1048576,
		MDecode = 1048575,
	};
	
	void getCutEdgeIndPos(int & cutInd,
				sdb::Array<sdb::Coord2, ICutEdge > & edgeMap,
				int & numCut,
				const RedBlueRefine & refiner,
				const int & iv0,
				const int & iv1,
				const DistanceNode * a,
				const DistanceNode * b);
				
	void cutEdges(RedBlueRefine & refiner,
				const ITetrahedron * t,
				const DistanceNode * tetn,
				sdb::Array<sdb::Coord2, ICutEdge > & edgeMap,
				int & numCut);
/// i face j vertex
/// decode index to field node or added cut
	const Vector3F & triangleP(const int & i, const int & j) const;			
	void dumpTriangleInd(sdb::Sequence<sdb::Coord3 > & faces);
	void countTriangleVertices(sdb::Array<int, int> & vertMap);
	void dumpVertex(sdb::Array<int, int> & vertMap);
	void calculateVertexNormal(sdb::Array<int, int> & vertMap);
	void dumpIndices(sdb::Array<int, int> & vertMap);
	
};

}
}