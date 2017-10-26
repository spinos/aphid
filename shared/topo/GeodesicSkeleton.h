/*
 *  GeodesicSkeleton.h
 *  
 *  skeleton extraction by geodesic distance level sets
 *  joints in pieces, one piece per path
 *  a piece can have a parent joint
 *  connect pieces by detecting path change in neighbors
 *
 *  Created by jian zhang on 10/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TOPO_GEODESIC_SKELETON_H
#define APH_TOPO_GEODESIC_SKELETON_H

#include "GeodesicRegion.h"

namespace aphid {

namespace topo {

class GeodesicSkeleton : public GeodesicRegion {

	int m_numJoints;
	
typedef boost::scoped_array<int> IntArrTyp;
	IntArrTyp m_pieceCounts;
	IntArrTyp m_pieceBegins;
	IntArrTyp m_pieceParent;

typedef boost::scoped_array<Vector3F> PntArrTyp;
	PntArrTyp m_jointPos;
	
public:
	GeodesicSkeleton();
	virtual ~GeodesicSkeleton();
	
	bool buildSkeleton(const float& unitD,
					const Vector3F* pos);
	void clearAllJoint();
	
	const int& numJoints() const;
	const Vector3F* jointPos() const;
	
protected:

	bool buildJoints(PathData* pds);
	void connectPieces();
/// global
	int getJointIndex(const int& pieceI, const int& jointJ) const;
	int getPieceVaryingJointIndex(const int& x) const;
	
private:
	
};

}

}
#endif
