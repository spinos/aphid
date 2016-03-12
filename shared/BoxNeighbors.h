/*
 *  KdNeighbors.h
 *  testntree
 *
 *  Created by jian zhang on 3/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BoundingBox.h>

namespace aphid {

class BoxNeighbors {
	
public:
	BoxNeighbors();
	void reset();
	
	void setOpposite(const BoundingBox & box, int axis, bool isHigh, int treeletIdx, int nodeIdx);
	
	void setNeighbor(const BoundingBox & box, int idx, int treeletIdx, int nodeIdx);
	
	bool isEmpty() const;
	
	int encodeTreeletNodeHash(int i, int s) const;
	
	void verbose() const;
	
	static bool AreNeighbors(int dir, const BoundingBox & a, const BoundingBox & b);
	static void DecodeTreeletNodeHash(const int & src, int rank, int & itreelet, int & inode);
	
/// 0 left 1 right 2 bottom 3 top 4 back 5 front
	BoundingBox _n[6];
	
};

}

