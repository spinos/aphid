/*
 *  BaseBinSplit.cpp
 *  testntree
 *
 *  Created by jian zhang on 3/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseBinSplit.h"

namespace aphid {

#define MMBINISFLATRATIO .1f
#define MMBINCUTOFFRATIO .2f

BaseBinSplit::BaseBinSplit() 
{}

BaseBinSplit::~BaseBinSplit() 
{}

void BaseBinSplit::splitSoftBinAlong(MinMaxBins * dst,
			const int & axis,
			GridClustering * grd, const BoundingBox & box)
{
	float geoRight, rightMost = box.getMin(axis);
	grd->begin();
	while (!grd->end() ) {
		const BoundingBox & primBox = grd->value()->m_box;
		
		dst->insertSplitPos(primBox.getMin(axis) );
		
		if(dst->isFull() ) return;
		
		geoRight = primBox.getMax(axis);
		if(rightMost < geoRight) rightMost = geoRight;
		
		grd->next();
	}
	dst->insertSplitPos(rightMost + m_bins[axis].delta() );
	//if(m_bins[axis].numSplits() < 4) {
	//	std::cout<<"\n\n waringing low splits\n\n";
	//	m_bins[axis].printSplitPos();
	//}
}

void BaseBinSplit::initEvents(const BoundingBox & b)
{
	for(int axis = 0; axis < 3; axis++) {
		if(isEmptyAlong(axis))
			continue;
	
		initEventsAlong(firstEventAlong(axis), b, axis);
	}
}

void BaseBinSplit::initEventsAlong(SplitEvent * e,
                    const BoundingBox & b, const int &axis)
{
	const int nb = m_bins[axis].numSplits();
	unsigned leftNumPrim, rightNumPrim;
	for(int i = 0; i < nb; i++) {
		m_bins[axis].getCounts(i, leftNumPrim, rightNumPrim);
			
		e[i].setBAP(b, axis, m_bins[axis].splitPos(i) );
		e[i].setLeftRightNumPrim(leftNumPrim, rightNumPrim);
	}
}

bool BaseBinSplit::cutoffEmptySpace(int & dst, const BoundingBox & bb, const float & minVol)
{
	int res = -1;
	float mxd = -1.f, dist;
	int isplit=-1;
	for(int axis = 0; axis < 3; axis++) {
		if(isEmptyAlong(axis)) 
		    continue;
		
		dist = m_bins[axis].leftEmptyDistance(isplit);
		if(dist > mxd) {
			res = axis * MMBINNSPLITLIMIT + isplit;
			mxd = dist;
		}
		
		dist = m_bins[axis].rightEmptyDistance(isplit);
		if(dist > mxd) {
			res = axis * MMBINNSPLITLIMIT + isplit;
			mxd = dist;
		}
	}
	
	if(mxd < 0.f) return false;
	
	float vol = mxd * bb.crossSectionArea(res/MMBINNSPLITLIMIT);
	if(vol > minVol) {
		if(0) {
		std::cout<<"\n\n empty ratio "<<vol/bb.volume()
		<<"\n cost "<<m_event[res].getCost()
		<<"\n idx "<<res;
		//m_event[res].verbose();
		}
		dst = res;
		return true;
	}
	
	return false;
}

void BaseBinSplit::splitAtLowestCost(const BoundingBox & b)
{
	m_bestEventIdx = 0;
	if(isEmptyAlong(0)
		&& isEmptyAlong(1)
		&& isEmptyAlong(2) ) {
/// no split, the best one will be empty
				return;
			}
			
	float lowest = 1e39f;
	for(int axis = 0; axis < 3; axis++) {
	    if(isEmptyAlong(axis))
			continue;
		
		for(int i = 1; i < m_bins[axis].numSplits()-1; ++i) {
			const SplitEvent * e = splitAt(axis, i);
			if(e->getCost() < lowest) {
				lowest = e->getCost();
				m_bestEventIdx = i + MMBINNSPLITLIMIT * axis;
			}
		}
	}

	if(m_event[m_bestEventIdx].hasBothSides() ) {
	int lc = 0;
	if(cutoffEmptySpace(lc, b, b.volume() * MMBINCUTOFFRATIO)) {
		    m_bestEventIdx = lc;
#if 0
			std::cout<<" cutoff at "
				<<lc/SplitEvent::NumEventPerDimension
				<<":"
				<<lc%SplitEvent::NumEventPerDimension;
#endif
		}
	}

	
#if 0
	std::cout<<"\n\n best split "<<m_bestEventIdx;
	m_event[m_bestEventIdx].verbose();
	m_bins[m_bestEventIdx/MMBINNSPLITLIMIT].verbose();

	if(m_event[m_bestEventIdx].isEmpty() ) {
		std::cout<<"\n warning empty split event "<<m_bestEventIdx;
		m_event[m_bestEventIdx].verbose();
		std::cout<<"\n show all splits";
		for(int axis = 0; axis < 3; axis++) {
			if(isEmptyAlong(axis)) {
				std::cout<<"\n axis "<<axis<<" is empty";
				continue;
			}
			
			std::cout<<"\n axis "<<axis<<" n split "<<m_bins[axis].numSplits();
				
			for(int i = 1; i < m_bins[axis].numSplits()-1; ++i) {
				std::cout<<"\n split "<<i;
				const SplitEvent * e = splitAt(axis, i);
				e->verbose();
			}
		}
	}
#endif
}

void BaseBinSplit::calcEvenBin(const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes,
			const BoundingBox & b)
{
	const float thre = b.getLongestDistance() * MMBINISFLATRATIO;
	for(int axis = 0; axis < 3; ++axis) {
		if(b.distance(axis) < thre) {
		    m_bins[axis].setFlat();	
			continue;
		}
			
		m_bins[axis].createEven(b.getMin(axis), b.getMax(axis));
		
		for(unsigned i = 0; i < nprim; i++) {
			const BoundingBox * primBox = primBoxes[*indices[i]];
			m_bins[axis].add(primBox->getMin(axis), primBox->getMax(axis));
		}
		
		m_bins[axis].scan();
	}
}

void BaseBinSplit::calcEvent(const BoundingBox & box,
			const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes)
{	
	for(int axis = 0; axis < 3; axis++) {
		if(isEmptyAlong(axis))
			continue;	
		updateEventBBoxAlong(box, axis, nprim, indices, primBoxes);
	}
}

void BaseBinSplit::calculateCosts(const BoundingBox & box)
{
	const float ba = box.area();
	const float bv = box.volume();
	for(int axis=0; axis<3; ++axis) {
	    if(isEmptyAlong(axis))
			continue;
			
		int lastOne = m_bins[axis].numSplits()-2;
/// skip ones on bound
		for(int i = 1; i < m_bins[axis].numSplits()-1; ++i) {
			firstEventAlong(axis)[i].calculateCost(ba, bv );
		}
	}
}

void BaseBinSplit::splitSoftBinAlong(MinMaxBins * dst, 
			const int & axis,
			const BoundingBox & box,
			const unsigned & nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes)
{
	float geoRight, rightMost = box.getMin(axis);
	for(unsigned i = 0; i < nprim; i++) {
		const BoundingBox * primBox = primBoxes[*indices[i]];
		dst->insertSplitPos(primBox->getMin(axis) );
		if(dst->isFull() ) return;
		
		geoRight = primBox->getMax(axis);
		if(rightMost < geoRight) rightMost = geoRight;
	}
	dst->insertSplitPos(rightMost + dst->delta() );
	//if(dst->numSplits() < 4) {
	//	std::cout<<"\n\n waringing low splits\n\n";
	//	dst->printSplitPos();
	//}
}

void BaseBinSplit::calcSoftBin(const unsigned & nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes,
			const BoundingBox & box)
{
	const float thre = box.getLongestDistance() * MMBINISFLATRATIO;
	for(int axis = 0; axis < 3; axis++) {
		if(box.distance(axis) < thre) {
		    m_bins[axis].setFlat();	
			continue;
		}
		
		m_bins[axis].reset(box.getMin(axis), box.getMax(axis) );
		
		splitSoftBinAlong(&m_bins[axis], axis, box, nprim, indices, primBoxes);
		
		for(unsigned i = 0; i < nprim; i++) {
			const BoundingBox * primBox = primBoxes[*indices[i]];
			m_bins[axis].add(primBox->getMin(axis), primBox->getMax(axis));
		}
		
		m_bins[axis].scan();
		//m_bins[axis].verbose();
	}
}

void BaseBinSplit::calcSoftBin(GridClustering * grd, const BoundingBox & box)
{
	const float thre = box.getLongestDistance() * MMBINISFLATRATIO;
	for(int axis = 0; axis < 3; axis++) {
		if(box.distance(axis) < thre) {
		    m_bins[axis].setFlat();
			continue;
		}
		
		m_bins[axis].reset(box.getMin(axis), box.getMax(axis) );
		
		splitSoftBinAlong(&m_bins[axis], axis, grd, box);
		
		grd->begin();
		while (!grd->end() ) {
			const BoundingBox & primBox = grd->value()->m_box;
			if(primBox.touch(box) ) 
				m_bins[axis].add(primBox.getMin(axis), primBox.getMax(axis));
				
			grd->next();
		}
		
		m_bins[axis].scan();
		//m_bins[axis].verbose();
	}
}

void BaseBinSplit::calcEvenBin(GridClustering * grd, const BoundingBox & b)
{
	const float thre = b.getLongestDistance() * MMBINISFLATRATIO;
	for(int axis = 0; axis < 3; axis++) {
		if(b.distance(axis) < thre) {
		    m_bins[axis].setFlat();	
			continue;
		}
			
		m_bins[axis].createEven(b.getMin(axis), b.getMax(axis));
		
		grd->begin();
		while (!grd->end() ) {
			const BoundingBox & primBox = grd->value()->m_box;
			if(primBox.touch(b) ) 
				m_bins[axis].add(primBox.getMin(axis), primBox.getMax(axis));
				
			grd->next();
		}
		
		m_bins[axis].scan();
		//m_bins[axis].verbose();
	}
}

void BaseBinSplit::calcEvent(GridClustering * grd, const BoundingBox & box)
{	
	for(int axis = 0; axis < 3; axis++) {
		if(isEmptyAlong(axis))
			continue;	
		updateEventBBoxAlong(axis, grd, box);
	}
}

void BaseBinSplit::updateEventBBoxAlong(const int &axis,
				GridClustering * grd, const BoundingBox & box)
{
	SplitEvent * eventOffset = firstEventAlong(axis);
	
	int g, minGrid, maxGrid;
	int n = m_bins[axis].numSplits();
	
	grd->begin();
	while (!grd->end() ) {
		const BoundingBox & primBox = grd->value()->m_box;
			
			minGrid = m_bins[axis].firstSplitToRight(primBox.getMin(axis) );
			for(g = minGrid; g < n; g++)
				eventOffset[g].updateLeftBox(primBox);
		
			maxGrid = m_bins[axis].lastSplitToLeft(primBox.getMax(axis) );
			for(g = maxGrid; g >= 0; g--)
				eventOffset[g].updateRightBox(primBox);
				
		grd->next();
	}
}

void BaseBinSplit::updateEventBBoxAlong(const BoundingBox & box,
			const int &axis, 
			const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes)
{
	SplitEvent * eventOffset = firstEventAlong(axis);
	
	int n = m_bins[axis].numSplits();
	int g, minGrid, maxGrid;
	unsigned i;
	for(i = 0; i < nprim; i++) {
		const BoundingBox * primBox = primBoxes[*indices[i] ];
		
		minGrid = m_bins[axis].firstSplitToRight(primBox->getMin(axis) );
			
		for(g = minGrid; g < n; g++)
			eventOffset[g].updateLeftBox(*primBox);
		
		maxGrid = m_bins[axis].lastSplitToLeft(primBox->getMax(axis) );
		for(g = maxGrid; g >= 0; g--)
			eventOffset[g].updateRightBox(*primBox);
	}
}

SplitEvent * BaseBinSplit::splitAt(int axis, int idx)
{ return &m_event[axis * MMBINNSPLITLIMIT + idx]; }

SplitEvent * BaseBinSplit::split(int idx)
{ return &m_event[idx]; }

SplitEvent * BaseBinSplit::firstEventAlong(const int & axis)
{ return &m_event[axis * MMBINNSPLITLIMIT]; }

bool BaseBinSplit::isEmptyAlong(const int & axis) const
{ 
    return m_bins[axis].isFlat() 
		|| m_bins[axis].isEmpty(); 
}

}