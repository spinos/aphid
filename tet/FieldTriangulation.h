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
#include "RedBlueRefine.h"

namespace ttg {

class FieldTriangulation : public AdaptiveBccField {
	
	aphid::sdb::Coord3 * m_triInds;
	aphid::Vector3F * m_cutPosBuf;
	int m_maxCutPosBuf, m_numAddedVert, m_numFrontTris;
	
public:
	FieldTriangulation();
	virtual ~FieldTriangulation();
	
	void triangulateFront();
	
	const int & numFrontTriangles() const;
	void getTriangleShape(aphid::cvx::Triangle & t, const int & i) const;	
	
	const int & numAddedVertices() const;
	const aphid::Vector3F & addedVertex(const int & i) const;
/// i face j vertex
/// decode index to field node or added cut
	const aphid::Vector3F & triangleP(const int & i, const int & j) const;

protected:
	
private:
	struct ICutEdge {
		aphid::sdb::Coord2 key;
		int ind;
	};
	
	enum CutIndMask {
		MEncode = 1048576,
		MDecode = 1048575,
	};
	
	void getCutEdgeIndPos(int & cutInd,
				aphid::sdb::Array<aphid::sdb::Coord2, ICutEdge > & edgeMap,
				int & numCut,
				const RedBlueRefine & refiner,
				const int & iv0,
				const int & iv1,
				const aphid::DistanceNode * a,
				const aphid::DistanceNode * b);
				
	void cutEdges(RedBlueRefine & refiner,
				const ITetrahedron * t,
				const aphid::DistanceNode * tetn,
				aphid::sdb::Array<aphid::sdb::Coord2, ICutEdge > & edgeMap,
				int & numCut);
		
};

}