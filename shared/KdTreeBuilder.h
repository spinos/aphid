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
#include <BuildKdTreeContext.h>
#include <MinMaxBins.h>

class KdTreeBuilder {
public:
	typedef Primitive* PrimitivePtr;

	KdTreeBuilder(BuildKdTreeContext &ctx);
	virtual ~KdTreeBuilder();
	
	void calculateSides();
	
	void partitionLeft(BuildKdTreeContext &ctx);
	void partitionRight(BuildKdTreeContext &ctx);
	
	const SplitEvent *bestSplit();
		
private:
	struct IndexLimit {
		IndexLimit() {low = high = -1;}
		int low, high;
	};
	typedef std::vector<IndexLimit> EmptySpace;
	
	void calculateBins();
	void calculateSplitEvents();
	void cutoffEmptySpace();
	unsigned numEvents() const;
	
	BoundingBox m_bbox;
	BuildKdTreeContext *m_context;
	MinMaxBins *m_bins;
	SplitEvent *m_event;
	char *m_primitiveClassification;
	unsigned m_numPrimitive;
	unsigned m_bestEventIdx;
};