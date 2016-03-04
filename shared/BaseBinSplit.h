/*
 *  BaseBinSplit.h
 *  testntree
 *
 *  Created by jian zhang on 3/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <SplitEvent.h>
#include <MinMaxBins.h>
#include <BoundingBox.h>
#include <VectorArray.h>

namespace aphid {

class BaseBinSplit {

public:
	BaseBinSplit();
	virtual ~BaseBinSplit();
	
protected:
	MinMaxBins * m_bins;
	SplitEvent * m_event;
    
protected:
	void initBins(const BoundingBox & b);
	void initEvents(const BoundingBox & b);
	bool byCutoffEmptySpace(int & dst, const BoundingBox & bb);
	SplitEvent * splitAt(int axis, int idx) const;
	int splitAtLowestCost();
	void calculateBins(const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes);
	void calculateSplitEvents(const BoundingBox & box,
			const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes);
			
private:
	void initEventsAlong(const BoundingBox & b, const int &axis);
	void updateEventBBoxAlong(const BoundingBox & box,
			const int &axis, const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes);
	
};

}