/*
 *  ADistanceField.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ADistanceField.h"

namespace aphid {

ADistanceField::ADistanceField()
{}

ADistanceField::~ADistanceField()
{}

void ADistanceField::nodeColor(Vector3F & dst, const DistanceNode & n,
						const float & scale) const
{
	if(n.stat == sdf::StUnknown)
		return dst.set(.3f, .3f, .3f);
	
	if(n.val > 0.f) {
		float r = MixClamp01F<float>(1.f, 0.f, n.val * scale);
		dst.set(r, 1.f - r, 0.f);
	}
	else {
		float b = MixClamp01F<float>(1.f, 0.f, -n.val * scale);
		dst.set(.5f * (1.f - b), 1.f - b, b);
	}
}

void ADistanceField::initNodes()
{
	const int n = numNodes();
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		d.val = 1e9f;
		d.label = sdf::StBackGround;
		d.stat = sdf::StUnknown;
	}
}

void ADistanceField::fastMarchingMethod()
{
/// heap of trial
	std::map<int, int> trials;
	const int n = numNodes();
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		if(d.stat == sdf::StKnown) {
			propagate(trials, i);
		}
	}
	
/// for each trial
	while (trials.size() > 0) {

/// A is first in trial		
		i = trials.begin()->first;
/// distance is known after propagation
		nodes()[i].stat = sdf::StKnown;
/// remove A from trial
		trials.erase(trials.begin() );
		
/// from A
		propagate(trials, i);
		
		//std::cout<<"\n trial n "<<trials.size();
		//std::cout.flush();
	}
}

/// A to B
void ADistanceField::propagate(std::map<int, int > & heap, 
												const int & i)
{
	const DistanceNode & A = nodes()[i];
	
/// for each neighbor of A
	const int endj = edgeBegins()[i+1];
	int vj, j = edgeBegins()[i];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IGraphEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == i)
			vj = eg.vi.y;
			
		DistanceNode & B = nodes()[vj];
		if(B.stat == sdf::StUnknown) {
		
/// min distance to B via A
/// need eikonal approximation here
			if(A.val + eg.len < B.val)
				B.val = A.val + eg.len;
				
/// add to trial
			heap[vj] = 0;
		}
	}
}

}