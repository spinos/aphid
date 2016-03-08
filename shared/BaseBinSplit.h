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
#include <GridClustering.h>

namespace aphid {

class BaseBinSplit {

	MinMaxBins m_bins[3];
	SplitEvent m_event[MMBINNSPLITLIMIT * 3];
    
public:
	BaseBinSplit();
	virtual ~BaseBinSplit();
	
protected:
	int m_bestEventIdx;
	
protected:
	void initEvents(const BoundingBox & b);
	SplitEvent * splitAt(int axis, int idx);
	SplitEvent * split(int idx);
	SplitEvent * firstEventAlong(const int & axis);
	void splitAtLowestCost(const BoundingBox & b);
	void calcEvenBin(const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes,
			const BoundingBox & b);
	void calculateSplitEvents(const BoundingBox & box,
			const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes);
	void calculateCosts(const BoundingBox & box);
	void calcEvenBin(GridClustering * grd, const BoundingBox & box);
	void calculateCompressSplitEvents(GridClustering * grd, const BoundingBox & box);
	
	void calcSoftBin(GridClustering * grd, const BoundingBox & box);
	void calcSoftBin(const unsigned & nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes,
			const BoundingBox & box);
	
private:
	void initEventsAlong(SplitEvent * e,
	        const BoundingBox & b, const int &axis);
	void updateEventBBoxAlong(const BoundingBox & box,
			const int &axis, const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes);
	void updateEventBBoxAlong(const int &axis,
			GridClustering * grd, const BoundingBox & box);
	void splitSoftBinAlong(MinMaxBins * dst,
			const int & axis,
			GridClustering * grd, const BoundingBox & box);
	void splitSoftBinAlong(MinMaxBins * dst, 
			const int & axis,
			const BoundingBox & box,
			const unsigned & nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes);
	bool cutoffEmptySpace(int & dst, const BoundingBox & bb, const float & minVol);
	bool isEmptyAlong(const int & axis) const;
	
};

}