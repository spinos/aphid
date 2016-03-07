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

public:
	BaseBinSplit();
	virtual ~BaseBinSplit();
	
protected:
	MinMaxBins m_bins[3];
	SplitEvent m_event[MMBINNSPLITLIMIT * 3];
    int m_bestEventIdx;
	
protected:
	void initEvents(const BoundingBox & b);
	SplitEvent * splitAt(int axis, int idx);
	void splitAtLowestCost(const BoundingBox & b);
	void calculateBins(const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes,
			const BoundingBox & b);
	void calculateSplitEvents(const BoundingBox & box,
			const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes);
	void calculateCosts(const BoundingBox & box);
	void calculateCompressBins(GridClustering * grd, const BoundingBox & box);
	void calculateCompressSplitEvents(GridClustering * grd, const BoundingBox & box);
	
	void calcCompressedSoftBin(GridClustering * grd, const BoundingBox & box);
	void calcSoftBin(const unsigned & nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes,
			const BoundingBox & box);
	
private:
	void initEventsAlong(const BoundingBox & b, const int &axis);
	void updateEventBBoxAlong(const BoundingBox & box,
			const int &axis, const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes);
	void updateCompressEventBBoxAlong(const int &axis,
			GridClustering * grd, const BoundingBox & box);
	void testCompressedSoftBinAlong(const int & axis,
			GridClustering * grd, const BoundingBox & box);
	void createSoftBin(MinMaxBins * dst, 
			const int & axis,
			const unsigned & nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes);
	bool cutoffEmptySpace(int & dst, const BoundingBox & bb, const float & minVol);
	
};

}