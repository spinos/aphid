/*
 *  PartitionBound.h
 *  kdtree
 *
 *  Created by jian zhang on 10/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BoundingBox.h>
#include <SplitCandidate.h>

class PartitionBound {
public:
	PartitionBound() {}
	
	const unsigned numPrimitive() const {
		return parentMax - parentMin;
	}
	
	BoundingBox bbox;
	unsigned parentMin, parentMax;
	unsigned childMin, childMax;
};