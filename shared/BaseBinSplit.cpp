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
{
	m_bins = new MinMaxBins[SplitEvent::Dimension];
	m_event = new SplitEvent[SplitEvent::NumEventPerDimension * SplitEvent::Dimension];
}

BaseBinSplit::~BaseBinSplit() 
{
	delete[] m_bins;
    delete[] m_event;
}

void BaseBinSplit::initBins(const BoundingBox & b)
{
	const float thre = b.getLongestDistance() * .05f;
	int axis;
	for(axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(b.distance(axis) < thre) {
		    m_bins[axis].setFlat();	
			continue;
		}
			
		m_bins[axis].create(SplitEvent::NumBinPerDimension, b.getMin(axis), b.getMax(axis));
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
	const float min = b.getMin(axis);
	const float delta = b.distance(axis) / SplitEvent::NumBinPerDimension;
	const int dimOffset = axis * SplitEvent::NumEventPerDimension;
	unsigned leftNumPrim, rightNumPrim;
	int i;
	for(i = 0; i < SplitEvent::NumEventPerDimension; i++) {
		SplitEvent & event = m_event[dimOffset + i];
		event.setBAP(b, axis, min + delta * (i + 1) );
		m_bins[axis].get(i, leftNumPrim, rightNumPrim);
		event.setLeftRightNumPrim(leftNumPrim, rightNumPrim);
	}
}

bool BaseBinSplit::byCutoffEmptySpace(int & dst, const BoundingBox & bb)
{
	int res = -1;
	float vol, area, emptyVolume = -1.f;
	const int minHead = 2;
	const int maxTail = SplitEvent::NumEventPerDimension - 3;
	const int midSect = SplitEvent::NumEventPerDimension / 2;
	int i, head, tail;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat() ) continue;
		
		area = bb.crossSectionArea(axis);
		
		head = 0;
		SplitEvent * cand = splitAt(axis, 0);
		if(cand->leftCount() == 0) {
			for(i = minHead; i < midSect; i++) {
				cand = splitAt(axis, i);
				if(cand->leftCount() == 0)
					head = i;
			}
			
			if(head > minHead) {
				vol = head * m_bins[axis].delta() * area;
				
				if(vol > emptyVolume) {
					emptyVolume = vol;
					res = SplitEvent::NumEventPerDimension * axis + head;
				}
			}
		}
		tail = SplitEvent::NumEventPerDimension - 1;
		cand = splitAt(axis, SplitEvent::NumEventPerDimension - 1);
		if(cand->rightCount() == 0) {
			for(i = maxTail; i > midSect; i--) {
				cand = splitAt(axis, i);
				if(cand->rightCount() == 0)
					tail = i;
			}
			
			if(tail < maxTail) {
				vol = (SplitEvent::NumEventPerDimension - tail) * m_bins[axis].delta() * area;
				
				if(vol > emptyVolume) {
					emptyVolume = vol;
					res = SplitEvent::NumEventPerDimension * axis + tail;
				}
			}
		}
	}
	if(res > 0) dst = res;
	
	return res>0;
}

SplitEvent * BaseBinSplit::splitAt(int axis, int idx) const
{ return &m_event[axis * SplitEvent::NumEventPerDimension + idx]; }

int BaseBinSplit::splitAtLowestCost()
{
	float lowest = 10e28f;
	int result = 0;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		for(int i = 1; i < SplitEvent::NumEventPerDimension - 1; i++) {
			const SplitEvent * e = splitAt(axis, i);
			if(e->getCost() < lowest && e->hasBothSides()) {
				lowest = e->getCost();
				result = i + SplitEvent::NumEventPerDimension * axis;
			}
		}
	}
    return result;
}

void BaseBinSplit::calculateBins(const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes)
{
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if( m_bins[axis].isFlat()) {
			continue;
		}
		
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
			
		updateEventBBoxAlong(box, axis, nprim, indices, primBoxes);
	}	
}

void BaseBinSplit::updateEventBBoxAlong(const BoundingBox & box,
			const int &axis, 
			const unsigned nprim, 
			const sdb::VectorArray<unsigned> & indices,
			const sdb::VectorArray<BoundingBox> & primBoxes)
{
	SplitEvent * eventOffset = &m_event[axis * SplitEvent::NumEventPerDimension];
	
	const float min = box.getMin(axis);
	const float delta = box.distance(axis) / SplitEvent::NumBinPerDimension;
	int g, minGrid, maxGrid;
	for(unsigned i = 0; i < nprim; i++) {
		const BoundingBox * primBox = primBoxes[*indices[i] ];
		
		minGrid = (primBox->getMin(axis) - min) / delta;
		
		if(minGrid < 0) minGrid = 0;
		
		for(g = minGrid; g < SplitEvent::NumEventPerDimension; g++)
			eventOffset[g].updateLeftBox(*primBox);
		
		maxGrid = (primBox->getMax(axis) - min) / delta;
		
		if(maxGrid > SplitEvent::NumEventPerDimension) maxGrid = SplitEvent::NumEventPerDimension;

		for(g = maxGrid; g > 0; g--)
			eventOffset[g - 1].updateRightBox(*primBox);
	}
}

}