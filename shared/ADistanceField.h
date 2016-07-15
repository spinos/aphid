/*
 *  ADistanceField.h
 *  
 *	distance field with function
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "AGraph.h"
#include "BDistanceFunction.h"
#include <map>

namespace aphid {

namespace sdf {
enum NodeState {
	StBackGround = 0,
	StFront = 1,
	StUnknown = 2,
	StKnown = 3
};

}

struct DistanceNode {
	
	Vector3F pos;
	float val;
	short label;
	short stat;
};
 
class ADistanceField : public AGraph<DistanceNode> {

public:
	ADistanceField();
	virtual ~ADistanceField();
	
	void nodeColor(Vector3F & dst, const DistanceNode & n,
					const float & scale) const;
	
protected:
	void initNodes();
	
	template<typename Tf>
	void calculateDistanceOnFront(const Tf * func)
	{
		const int n = numNodes();
		int i = 0;
		for(;i<n;++i) {
			DistanceNode * d = &nodes()[i];
			if(d->label == sdf::StFront) {
				d->val = func->calculateDistance(d->pos);
				d->stat = sdf::StKnown; /// accept
			}
		}
	}
	
/// propagate distance value	
	void fastMarchingMethod();
	
private:
	void propagate(std::map<int, int > & heap, const int & i);
	
};

}