/*
 *  SparseVoxelOctree.cpp
 *  
 *
 *  Created by jian zhang on 2/15/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "SparseVoxelOctree.h"

namespace aphid {

namespace ttg {

void SVOBNode::setChildKey(const int& x, const int& i)
{ _ind[i+1] = x; }
	
void SVOBNode::setLeaf()
{ memset(&_ind[1], 0, 32); }
	
bool SVOBNode::isLeaf() const 
{
	for(int i=1;i<9;++i) {
		if(_ind[i] > 0)
			return false;
	}
	return true;
}
	
void SVOBNode::connectToParent(const int& pid, const int& i)
{
	_ind[0] = (i<<27) | pid;
}

int SVOBNode::parentInd() const
{ return _ind[0] & (1<<27)-1; }

void SVOBNode::getChildInds(int* dst) const
{
	memcpy(dst, &_ind[1], 32);
}

void SVOBNode::getParent(int& dst) const
{
	dst = _ind[0];
}

void SVOBNode::getLocation(int& dst) const
{
	dst = _ind[9];
}

	
}

}