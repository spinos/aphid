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
#include <BaseBinSplit.h>
#include <BuildKdTreeContext.h>

namespace aphid {

class KdTreeBuilder : public BaseBinSplit {
public:
	typedef Primitive* PrimitivePtr;

	KdTreeBuilder();
	virtual ~KdTreeBuilder();
	
	void setContext(BuildKdTreeContext &ctx);
	void calculateSides();
	
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
	
	void partitionCompress(const SplitEvent & e,
						const BoundingBox & leftBox, const BoundingBox & rightBox,
						BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx);
	void partitionPrims(const SplitEvent & e,
						const BoundingBox & leftBox, const BoundingBox & rightBox,
						BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx);
	
	BoundingBox m_bbox;
	BuildKdTreeContext *m_context;
};

}