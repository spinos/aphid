/*
 *  BoxNeighbors.cpp
 *  testntree
 *
 *  Created by jian zhang on 3/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BoxNeighbors.h"
#include <iostream>
#include <AllMath.h>

namespace aphid {

BoxNeighbors::BoxNeighbors() {}

void BoxNeighbors::reset() 
{
	int i = 0;
	for(;i<6;i++) {
		_n[i].m_padding0 = 0; // node
		_n[i].m_padding1 = 0; // treelet, zero is null
	}
}
	
void BoxNeighbors::setOpposite(const BoundingBox & box, int axis, bool isHigh, int treeletIdx, int nodeIdx)
{
	int idx = axis<<1;
	if(isHigh) idx++;
	setNeighbor(box, idx, treeletIdx, nodeIdx);
}

void BoxNeighbors::setNeighbor(const BoundingBox & box, int idx, int treeletIdx, int nodeIdx)
{
	_n[idx] = box;
	_n[idx].m_padding0 = nodeIdx;
	_n[idx].m_padding1 = treeletIdx;
}

bool BoxNeighbors::isEmpty() const
{
	int i = 0;
	for(;i<6;i++) {
		if(_n[i].m_padding1 > 0) return false;
	}
	return true;
}

int BoxNeighbors::encodeTreeletNodeHash(int i, int s) const
{ return (_n[i].m_padding1 << (s + 1)) | _n[i].m_padding0; }

void BoxNeighbors::DecodeTreeletNodeHash(const int & src, int rank, int & itreelet, int & inode)
{
	itreelet = src >> (rank+1);
	inode = src & ((1<<(rank+1))-1 );
}

void BoxNeighbors::verbose() const
{
	std::cout<<"\n BoxNeighbors";
	int i = 0;
	for(;i<6;i++) {
		if(_n[i].m_padding1 != 0) std::cout<<"\n ["<<i<<"] "<<_n[i].m_padding1
			<<" "<<_n[i].m_padding0
			<<" "<<_n[i];
	}
}

bool BoxNeighbors::IsNeighborOf(int dir, const BoundingBox & a, const BoundingBox & b,
								const float & tolerance)
{
	const int splitAxis = dir / 2;
	int i = 0;
	for(;i<3;i++) {
		if(i==splitAxis) {
			if(dir & 1) {
				if(Absolute<float>(b.getMin(splitAxis) - a.getMax(splitAxis) ) > tolerance ) return false;
			}
			else {
				if(Absolute<float>(b.getMax(splitAxis) - a.getMin(splitAxis) ) > tolerance ) return false;
			}
		}
		else {
			if(b.getMin(i) > a.getMin(i) ) return false;
			if(b.getMax(i) < a.getMax(i) ) return false;
		}
	}
	return true;
}

}