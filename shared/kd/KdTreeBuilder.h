/*
 *  KdTreeBuilder.h
 *  kdtree
 *
 *  Created by jian zhang on 10/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <kd/BaseBinSplit.h>
#include <kd/BuildKdTreeContext.h>

namespace aphid {

class KdTreeBuilder : public BaseBinSplit {
	
	BoundingBox m_bbox;
	BuildKdTreeContext *m_context;
	
public:
	KdTreeBuilder();
	virtual ~KdTreeBuilder();
	
	void setContext(BuildKdTreeContext &ctx);
	void calculateSides();
	
	void partition(BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx);
	
	const SplitEvent *bestSplit();
	
	static int MaxLeafPrimThreashold;
	static int MaxBuildLevel;
	
private:
	void partitionCompress(const SplitEvent & e,
						const BoundingBox & leftBox, const BoundingBox & rightBox,
						BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx);
	void partitionPrims(const SplitEvent & e,
						const BoundingBox & leftBox, const BoundingBox & rightBox,
						BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx);
						
};

}