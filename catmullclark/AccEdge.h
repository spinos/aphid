/*
 *  AccEdge.h
 *  knitfabric
 *
 *  Created by jian zhang on 5/30/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <Vector3F.h>

namespace aphid {

class AccEdge {
public:
	AccEdge() {}
	void reset();
	Vector3F computePosition(int side) const;
	Vector3F computeNormal(int side) const;
	
	Vector3F _edgePositions[2];
	Vector3F _edgeNormals[2];
	Vector3F _fringePositions[4];
	Vector3F _fringeNormals[4];
	float _valence[2];
	char _isZeroLength;
	char _isBoundary;
private:

};

}