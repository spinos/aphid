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

class JointPiece;

class GeodesicSkeleton : public GeodesicRegion {

	boost::scoped_array<JointPiece > m_pieces;
	int m_numPieces;
		
public:
	GeodesicSkeleton();
	virtual ~GeodesicSkeleton();
	
	bool buildSkeleton();
	void clearAllJoint();
	
	const int& numPieces() const;
	const JointPiece& getPiece(const int& i) const;	
	
protected:

	//void connectPieces();
/// global
	int getJointIndex(const int& pieceI, const int& jointJ) const;
	int getPieceVaryingJointIndex(const int& x) const;
	
private:
	void buildClusters(const std::vector<int >& vertexSet,
						const int& jregion);
	
};

}

}
#endif
