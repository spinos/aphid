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

BaseBinSplit::BaseBinSplit() 
{}

BaseBinSplit::~BaseBinSplit() 
{}

void BaseBinSplit::splitSoftBinAlong(const int & axis,
			GridClustering * grd, const BoundingBox & box)
{
	float geoRight, rightMost = box.getMin(axis);
	grd->begin();
	while (!grd->end() ) {
		const BoundingBox & primBox = grd->value()->m_box;
		
		m_bins[axis].insertSplitPos(primBox.getMin(axis) );
		
		if(m_bins[axis].isFull() ) return;
		
		geoRight = primBox.getMax(axis);
		if(rightMost < geoRight) rightMost = geoRight;
		
		grd->next();
	}
	m_bins[axis].insertSplitPos(rightMost + m_bins[axis].delta() );
	if(m_bins[axis].numSplits() < 4) {
		std::cout<<"\n\n waringing low splits\n\n";
		m_bins[axis].printSplitPos();
	}
}

void BaseBinSplit::initEvents(const BoundingBox & b)
{
	int axis;
    for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
	
		initEventsAlong(b, axis);
	}
}

void BaseBinSplit::initEventsAlong(const BoundingBox & b, const int &axis)
{
	//std::cout<<"\n init event axis"<<axis
	//<<" n split "<<m_bins[axis].numSplits();
	//const float min = b.getMin(axis);
	//const float delta = b.distance(axis) / SplitEvent::NumBinPerDimension;
	const int dimOffset = axis * MMBINNSPLITLIMIT;
	int nb = m_bins[axis].numSplits();
	unsigned leftNumPrim, rightNumPrim;
	int i;
	// for(i = 0; i < SplitEvent::NumEventPerDimension; i++) {
	for(i = 0; i < nb; i++) {
		m_bins[axis].getCounts(i, leftNumPrim, rightNumPrim);
		
		//std::cout<<"\n p"<<m_bins[axis].splitPos(i)
			//<<" lft/rgt "<<leftNumPrim<<"/"<<rightNumPrim;
			
		SplitEvent & event = m_event[dimOffset + i];
		event.setBAP(b, axis, m_bins[axis].splitPos(i) );
		// event.setBAP(b, axis, min + delta * (i + 1) );
		event.setLeftRightNumPrim(leftNumPrim, rightNumPrim);
		//std::cout<<"\n event"<<i+dimOffset;
		//event.verbose();
	}
}

bool BaseBinSplit::cutoffEmptySpace(int & dst, const BoundingBox & bb, const float & minVol)
{
	int res = -1;
	float mxd = -1.f, dist;
	int isplit=-1;
	//int axis = bb.getLongestAxis();
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat() ) continue;
		
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
	float lowest = 1e28f;
	m_bestEventIdx = 0;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		for(int i = 0; i < MMBINNSPLITLIMIT; i++) {
			const SplitEvent * e = splitAt(axis, i);
			if(e->getCost() < lowest 
				&& e->hasBothSides()
				) {
				lowest = e->getCost();
				m_bestEventIdx = i + MMBINNSPLITLIMIT * axis;
			}
		}
	}
	//std::cout<<"\n lowest cost "<<lowest;
	//m_event[m_bestEventIdx].verbose();
#if 1
	int lc = 0;
	if(cutoffEmptySpace(lc, b, b.volume() * .34f)) {
		// if(m_event[lc].getCost() < lowest * 2.f )
		    m_bestEventIdx = lc;
#if 0
			std::cout<<" cutoff at "
				<<lc/SplitEvent::NumEventPerDimension
				<<":"
				<<lc%SplitEvent::NumEventPerDimension;
#endif
	}
#endif
	//if(!m_event[m_bestEventIdx].hasBothSides()) {
	//if(m_bins[m_bestEventIdx/MMBINNSPLITLIMIT].numSplits() < 4) {
	if(0) {
		std::cout<<"\n\n best split "<<m_bestEventIdx;
		m_event[m_bestEventIdx].verbose();
		m_bins[m_bestEventIdx/MMBINNSPLITLIMIT].verbose();
	}
}

void BaseBinSplit::calculateBins(const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes,
			const BoundingBox & b)
{
	const float thre = b.getLongestDistance() * .1f;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
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

void BaseBinSplit::calculateSplitEvents(const BoundingBox & box,
			const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes)
{	
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		if(m_bins[axis].isEmpty())
			continue;	
		updateEventBBoxAlong(box, axis, nprim, indices, primBoxes);
	}
}

void BaseBinSplit::calculateCosts(const BoundingBox & box)
{
	const float ba = box.area();
	for(int j=0; j<3; ++j) {
/// skip ones on bound
		for(int i = 1; i < m_bins[j].numSplits()-1; ++i) {
			m_event[i + j * MMBINNSPLITLIMIT].calculateCost(ba);
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
	if(dst->numSplits() < 4) {
		std::cout<<"\n\n waringing low splits\n\n";
		dst->printSplitPos();
	}
}

void BaseBinSplit::calcSoftBin(const unsigned & nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes,
			const BoundingBox & box)
{
	const float thre = box.getLongestDistance() * .1f;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(box.distance(axis) < thre) {
		    m_bins[axis].setFlat();	
			continue;
		}
		
		m_bins[axis].reset(box.getMin(axis), box.getMax(axis) );
		
		splitSoftBinAlong(&m_bins[axis], axis, box, nprim, indices, primBoxes);
		
		//std::cout<<"\n n prim"<<nprim;
		//m_bins[axis].printSplitPos();
		for(unsigned i = 0; i < nprim; i++) {
			const BoundingBox * primBox = primBoxes[*indices[i]];
			m_bins[axis].add(primBox->getMin(axis), primBox->getMax(axis));
		}
		
		m_bins[axis].scan();
		//m_bins[axis].verbose();
	}
}

void BaseBinSplit::calcCompressedSoftBin(GridClustering * grd, const BoundingBox & box)
{
	const float thre = box.getLongestDistance() * .1f;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(box.distance(axis) < thre) {
		    m_bins[axis].setFlat();	
			continue;
		}
		
		m_bins[axis].reset(box.getMin(axis), box.getMax(axis) );
		
		splitSoftBinAlong(axis, grd, box);
		
		// m_bins[axis].printSplitPos();
		
		grd->begin();
		while (!grd->end() ) {
			const BoundingBox & primBox = grd->value()->m_box;
			if(primBox.touch(box) ) 
				m_bins[axis].add(primBox.getMin(axis), primBox.getMax(axis));
				
			grd->next();
		}
		
		//if(axis ==0) std::cout<<"\n\n axis "<<axis;
		m_bins[axis].scan();
		//m_bins[axis].verbose();
		//std::cout<<"\n bb"<<box;
	}
}

void BaseBinSplit::calculateCompressBins(GridClustering * grd, const BoundingBox & b)
{
	const float thre = b.getLongestDistance() * .1f;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
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
		//m_bins[axis].printSplitPos();
		//m_bins[axis].verbose();
	}
}

void BaseBinSplit::calculateCompressSplitEvents(GridClustering * grd, const BoundingBox & box)
{	
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		if(m_bins[axis].isEmpty())
			continue;	
		updateEventBBoxAlong(axis, grd, box);
	}
}

void BaseBinSplit::updateEventBBoxAlong(const int &axis,
				GridClustering * grd, const BoundingBox & box)
{
	SplitEvent * eventOffset = &m_event[axis * MMBINNSPLITLIMIT];
	
	//const float min = box.getMin(axis);
	//const float delta = box.distance(axis) / SplitEvent::NumBinPerDimension;
	int g, minGrid, maxGrid;
	int n = m_bins[axis].numSplits();
	
	// m_bins[axis].printSplitPos();
	
	grd->begin();
	while (!grd->end() ) {
		const BoundingBox & primBox = grd->value()->m_box;
		//if(primBox.touch(box) ) {	
			//minGrid = (primBox.getMin(axis) - min) / delta;
		
			//if(minGrid < 0) minGrid = 0;
			//std::cout<<"\n t"<<primBox.getMin(axis);
		
			minGrid = m_bins[axis].firstSplitToRight(primBox.getMin(axis) );
			for(g = minGrid; g < n; g++)
				eventOffset[g].updateLeftBox(primBox);
		
			//maxGrid = (primBox.getMax(axis) - min) / delta;
		
			//if(maxGrid > SplitEvent::NumEventPerDimension) maxGrid = SplitEvent::NumEventPerDimension;
			maxGrid = m_bins[axis].lastSplitToLeft(primBox.getMax(axis) );
			if(maxGrid>0) maxGrid--;
			for(g = maxGrid; g >= 0; g--)
				eventOffset[g].updateRightBox(primBox);
				
			//std::cout<<" grid "<<minGrid<<"/"<<maxGrid;
		//}
		grd->next();
	}
}

void BaseBinSplit::updateEventBBoxAlong(const BoundingBox & box,
			const int &axis, 
			const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes)
{
	SplitEvent * eventOffset = &m_event[axis * MMBINNSPLITLIMIT];
	
	//const float min = box.getMin(axis);
	//const float delta = box.distance(axis) / SplitEvent::NumBinPerDimension;
	int n = m_bins[axis].numSplits();
	int g, minGrid, maxGrid;
	unsigned i;
	for(i = 0; i < nprim; i++) {
		const BoundingBox * primBox = primBoxes[*indices[i] ];
		
		//minGrid = (primBox->getMin(axis) - min) / delta;
		
		//if(minGrid < 0) minGrid = 0;
		minGrid = m_bins[axis].firstSplitToRight(primBox->getMin(axis) );
			
		for(g = minGrid; g < n; g++)
			eventOffset[g].updateLeftBox(*primBox);
		
		//maxGrid = (primBox->getMax(axis) - min) / delta;
		//if(maxGrid > SplitEvent::NumEventPerDimension) maxGrid = SplitEvent::NumEventPerDimension;
		maxGrid = m_bins[axis].lastSplitToLeft(primBox->getMax(axis) );
		if(maxGrid>0) maxGrid--;
		for(g = maxGrid; g >= 0; g--)
			eventOffset[g].updateRightBox(*primBox);
	}
}

SplitEvent * BaseBinSplit::splitAt(int axis, int idx)
{ return &m_event[axis * MMBINNSPLITLIMIT + idx]; }

}