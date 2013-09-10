/*
 *  KdTreeBuilder.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "KdTreeBuilder.h"
#include <boost/thread.hpp>  
#define NUMEVENTTHREAD 96

KdTreeBuilder::KdTreeBuilder(BuildKdTreeContext &ctx) 
{
	m_context = &ctx;
	m_numPrimitive = ctx.getNumPrimitives();
	m_bbox = ctx.getBBox();
	
	calculateBins();
	calculateSplitEvents();
}

KdTreeBuilder::~KdTreeBuilder() 
{
	//printf("builder quit\n");
	delete[] m_event;
	delete[] m_bins;
}

void KdTreeBuilder::calculateBins()
{
	BoundingBox *primBoxes = m_context->boxes();
	m_bins = new MinMaxBins[SplitEvent::Dimension];
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		//printf("bbox size %f\n", m_bbox.getMax(axis) - m_bbox.getMin(axis));
		if(m_bbox.distance(axis) < 10e-4) {
		    //printf("bbox[%i] is flat", axis);
			m_bins[axis].setFlat();
			continue;
		}
		m_bins[axis].create(SplitEvent::NumBinPerDimension, m_bbox.getMin(axis), m_bbox.getMax(axis));
	
		for(unsigned i = 0; i < m_numPrimitive; i++) {
			BoundingBox &primBox = primBoxes[i];
			m_bins[axis].add(primBox.getMin(axis), primBox.getMax(axis));
		}
		
		m_bins[axis].scan();
	}
}

void KdTreeBuilder::calculateSplitEvents()
{
	//SplitEvent::ParentBoxArea = m_bbox.area();
	//SplitEvent::ParentBox = m_bbox;
	
	const unsigned numEvent = SplitEvent::NumEventPerDimension * SplitEvent::Dimension;
	m_event = new SplitEvent[numEvent];
	int dimOffset;
	unsigned leftNumPrim, rightNumPrim;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		dimOffset = SplitEvent::NumEventPerDimension * axis;	
		const float min = m_bbox.getMin(axis);
		const float delta = m_bbox.distance(axis) / SplitEvent::NumBinPerDimension;
		for(int i = 0; i < SplitEvent::NumEventPerDimension; i++) {
			SplitEvent &event = m_event[dimOffset + i];
			event.setAxis(axis);
			event.setPos(min + delta * (i + 1));
			m_bins[axis].get(i, leftNumPrim, rightNumPrim);
			event.setLeftRightNumPrim(leftNumPrim, rightNumPrim);
		}
	}
/*	
	boost::thread eventThread[NUMEVENTTHREAD];
	
	for(unsigned i = 0; i < NUMEVENTTHREAD; i++) {
		eventThread[i] = boost::thread(&SplitEvent::calculateCost, &m_event[i]);
	}
	
	for(unsigned i = 0; i < NUMEVENTTHREAD; i++) {
		eventThread[i].join();
	}
	*/

	boost::thread boxThread[3];
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis] = boost::thread(boost::bind(&KdTreeBuilder::updateEventBBoxAlong, this, axis));
	}
	
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
		boxThread[axis].join();
	}
	
	for(unsigned i = 0; i < numEvent; i++) {
		m_event[i].calculateCost(m_bbox.area());
	}
}

void KdTreeBuilder::updateEventBBoxAlong(const int &axis)
{
	BoundingBox *primBoxes = m_context->boxes();
	for(unsigned i = 0; i < m_numPrimitive; i++) {
		const BoundingBox &primBox = primBoxes[i];
		
		const float min = m_bbox.getMin(axis);
		const float delta = m_bbox.distance(axis) / SplitEvent::NumBinPerDimension;
		const int eventOffset = axis * SplitEvent::NumEventPerDimension;
		
		int minGrid = (primBox.getMin(axis) - min) / delta;
		
		if(minGrid < 0) minGrid = 0;
		
		for(int g = minGrid; g < SplitEvent::NumEventPerDimension; g++)
			m_event[eventOffset + g].updateLeftBox(primBox);

		int maxGrid = (primBox.getMax(axis) - min) / delta;
		
		if(maxGrid > SplitEvent::NumBinPerDimension) maxGrid = SplitEvent::NumBinPerDimension;

		for(int g = maxGrid; g > 0; g--)
			m_event[eventOffset + g - 1].updateRightBox(primBox);
	}
}

const SplitEvent *KdTreeBuilder::bestSplit()
{
	m_bestEventIdx = 0;
	byLowestCost(m_bestEventIdx);
	
	unsigned lc = 0;
	if(byCutoffEmptySpace(lc)) {
		if(m_event[lc].getCost() < m_event[m_bestEventIdx].getCost() * 2.f)
			m_bestEventIdx = lc;
	}
		
	return &m_event[m_bestEventIdx];
}

SplitEvent KdTreeBuilder::splitAt(int axis, int idx) const
{
	return m_event[axis * SplitEvent::NumEventPerDimension + idx];
}

void KdTreeBuilder::byLowestCost(unsigned & dst)
{
	float lowest = 10e28;
	
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		for(int i = 1; i < SplitEvent::NumEventPerDimension - 1; i++) {
			SplitEvent e = splitAt(axis, i);
			if(e.getCost() < lowest && e.hasBothSides()) {
				lowest = e.getCost();
				dst = i + SplitEvent::NumEventPerDimension * axis;
			}
		}
	}
}

char KdTreeBuilder::byCutoffEmptySpace(unsigned &dst)
{
	int res = -1;
	float vol, emptyVolume = -1.f;
	int i, head, tail;
	SplitEvent cand;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		head = 0;
		cand = splitAt(axis, 0);
		if(cand.leftCount() == 0) {
			for(i = 1; i < SplitEvent::NumEventPerDimension - 1; i++) {
				cand = splitAt(axis, i);
				if(cand.leftCount() == 0)
					head = i;
			}
			
			if(head > 2) {
				vol = head;
				if(vol > emptyVolume) {
					emptyVolume = vol;
					res = SplitEvent::NumEventPerDimension * axis + head;
				}
			}
		}
		tail = SplitEvent::NumEventPerDimension - 1;
		cand = splitAt(axis, SplitEvent::NumEventPerDimension - 1);
		if(cand.rightCount() == 0) {
			for(i = 1; i < SplitEvent::NumEventPerDimension - 1; i++) {
				cand = splitAt(axis, SplitEvent::NumEventPerDimension - 1 - i);
				if(cand.rightCount() == 0)
					tail = SplitEvent::NumEventPerDimension - 1 - i;
			}
			if(tail < SplitEvent::NumEventPerDimension - 3) {
				vol = SplitEvent::NumEventPerDimension - tail;
				if(vol > emptyVolume) {
					emptyVolume = vol;
					res = SplitEvent::NumEventPerDimension * axis + tail;
				}
			}
		}
	}
	if(res > 0) {
		dst = res;
		//printf("cutoff at %i: %i left %i right %i\n", res/SplitEvent::NumEventPerDimension,  res%SplitEvent::NumEventPerDimension, m_event[res].leftCount(), m_event[res].rightCount());
	}
	return res>0;
}

void KdTreeBuilder::partitionLeft(BuildKdTreeContext &ctx)
{	
	SplitEvent &e = m_event[m_bestEventIdx];
	if(e.leftCount() < 1) return;
	ctx.create(e.leftCount());
	
	BoundingBox leftBox, rightBox;

	m_bbox.split(e.getAxis(), e.getPos(), leftBox, rightBox);
	ctx.setBBox(leftBox);
	
	unsigned *indices = m_context->indices();
	BoundingBox *boxSrc = m_context->boxes();
	BoundingBox *boxDst = ctx.boxes();
	unsigned *idxDst = ctx.indices();
	int count = 0;
	for(unsigned i = 0; i < m_numPrimitive; i++) {
		BoundingBox &primBox = boxSrc[i];
		int side = e.side(primBox);
		if(side < 2) {
			idxDst[count] = indices[i];
			boxDst[count] = boxSrc[i];
			count++;
		}
	}
	printf("%i to left side\n", count);	
}

void KdTreeBuilder::partitionRight(BuildKdTreeContext &ctx)
{	
	SplitEvent &e = m_event[m_bestEventIdx];
	if(e.rightCount() < 1) return;
	
	ctx.create(e.rightCount());
	
	BoundingBox leftBox, rightBox;

	m_bbox.split(e.getAxis(), e.getPos(), leftBox, rightBox);
	ctx.setBBox(rightBox);
	
	unsigned *indices = m_context->indices();
	BoundingBox *boxSrc = m_context->boxes();
	BoundingBox *boxDst = ctx.boxes();
	unsigned *idxDst = ctx.indices();
	int count = 0;
	for(unsigned i = 0; i < m_numPrimitive; i++) {
		BoundingBox &primBox = boxSrc[i];
		int side = e.side(primBox);
		if(side > 0) {
			idxDst[count] = indices[i];
			boxDst[count] = boxSrc[i];
			count++;
		}
	}
	printf("%i to right side\n", count);
}

void KdTreeBuilder::partition(BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx)
{
	
	SplitEvent &e = m_event[m_bestEventIdx];
	if(e.leftCount() > 0)
		leftCtx.create(e.leftCount());
	if(e.rightCount() > 0)
		rightCtx.create(e.rightCount());
	
	BoundingBox leftBox, rightBox;

	m_bbox.split(e.getAxis(), e.getPos(), leftBox, rightBox);
	leftCtx.setBBox(leftBox);
	rightCtx.setBBox(rightBox);
	
	unsigned *indices = m_context->indices();
	BoundingBox *boxSrc = m_context->boxes();
	BoundingBox *leftBoxDst = leftCtx.boxes();
	BoundingBox *rightBoxDst = rightCtx.boxes();
	unsigned *leftIdxDst = leftCtx.indices();
	unsigned *rightIdxDst = rightCtx.indices();
	
	int leftCount = 0;
	int rightCount = 0;
	int side;
	for(unsigned i = 0; i < m_numPrimitive; i++) {
		BoundingBox &primBox = boxSrc[i];
		
		//if(primBox.getMax(e.getAxis()) < m_bbox.getMin(e.getAxis())) continue;
		//if(primBox.getMin(e.getAxis()) > m_bbox.getMax(e.getAxis())) continue;
		//if(*indices == 2202) printf("2202 xbound %f %f", primBox.getMin(0), primBox.getMax(0));
		side = e.side(primBox);
		
		//side = m_primitiveClassification[i];
		if(side < 2) {
			if(primBox.touch(leftBox)) {
		
			leftIdxDst[leftCount] = *indices;
			leftBoxDst[leftCount] = boxSrc[i];
			leftCount++;
			
			}
		}
		if(side > 0) {
			if(primBox.touch(rightBox)) {
			rightIdxDst[rightCount] = *indices;
			rightBoxDst[rightCount] = boxSrc[i];
			rightCount++;
			}
		}
		indices++;
	}
	//printf("partition %i | %i\n", leftCount, rightCount);
}

void KdTreeBuilder::verbose() const
{
	printf("unsplit cost %f = 2 * %i box %f\n", 2.f * m_numPrimitive, m_numPrimitive, m_bbox.area());
	m_event[m_bestEventIdx].verbose();
	printf("chose split %i: %i\n", m_bestEventIdx/SplitEvent::NumEventPerDimension,  m_bestEventIdx%SplitEvent::NumEventPerDimension);
}
