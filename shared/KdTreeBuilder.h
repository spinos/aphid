/*
 *  KdTreeBuilder.h
 *  kdtree
 *
 *  Created by jian zhang on 10/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

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
	void updateEventBBoxAlong(const int &axis);
	
	void partitionLeft(BuildKdTreeContext &ctx);
	void partitionRight(BuildKdTreeContext &ctx);
	void partition(BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx);
	
	const SplitEvent *bestSplit();
	void verbose() const;
private:
	struct IndexLimit {
		IndexLimit() {low = 1; high = -1;}
		char isValid() {return low < high;}
		int bound() {return high - low;}
		int low, high;
		float area;
	};
	typedef std::vector<IndexLimit> EmptySpace;
	
	void calculateBins();
	void calculateSplitEvents();
	char byCutoffEmptySpace(unsigned & dst);
	void byLowestCost(unsigned & dst);
	SplitEvent splitAt(int axis, int idx) const;
	
	BoundingBox m_bbox;
	BuildKdTreeContext *m_context;
	MinMaxBins *m_bins;
	SplitEvent *m_event;
	unsigned m_numPrimitive;
	unsigned m_bestEventIdx;
};