/*
 *  KdTreeBuilder.h
 *  kdtree
 *
 *  Created by jian zhang on 10/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <PartitionBound.h>
#include <PrimitiveArray.h>
#include <SplitEvent.h>

class KdTreeBuilder {
public:
	KdTreeBuilder();
	virtual ~KdTreeBuilder();
	
	void calculateSplitEvents(const PartitionBound &bound);
	
	const SplitEvent *bestSplit() const;
		
private:
	SplitEvent m_event;
};