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

namespace aphid {

KdTreeBuilder::KdTreeBuilder()
{
	m_bins = new MinMaxBins[SplitEvent::Dimension];
	m_event = new SplitEvent[SplitEvent::NumEventPerDimension * SplitEvent::Dimension];
}

KdTreeBuilder::~KdTreeBuilder() 
{
	delete[] m_event;
	delete[] m_bins;
}

void KdTreeBuilder::setContext(BuildKdTreeContext &ctx) 
{
	m_context = &ctx;
	m_bbox = ctx.getBBox();
	
	if(m_context->isCompressed() ) {
		calculateCompressBins();
		calculateCompressSplitEvents();
	}
	else {
		calculateBins();
		calculateSplitEvents();
	}
	
	const unsigned numEvent = SplitEvent::NumEventPerDimension * SplitEvent::Dimension;
	for(unsigned i = 0; i < numEvent; i++)
		m_event[i].calculateCost(m_bbox.area());
}

void KdTreeBuilder::calculateCompressBins()
{
	sdb::WorldGrid<GroupCell, unsigned > * grd = m_context->grid();
	
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bbox.distance(axis) < 1e-3f) {
		    m_bins[axis].setFlat();
			continue;
		}
		m_bins[axis].create(SplitEvent::NumBinPerDimension, m_bbox.getMin(axis), m_bbox.getMax(axis));
	
		grd->begin();
		while (!grd->end() ) {
			const BoundingBox & primBox = grd->value()->m_box;
			if(primBox.touch(m_bbox) ) 
				m_bins[axis].add(primBox.getMin(axis), primBox.getMax(axis));
				
			grd->next();
		}
		
		m_bins[axis].scan();
	}
}

void KdTreeBuilder::calculateCompressSplitEvents()
{
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
	
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
			
		updateCompressEventBBoxAlong(axis);
	}
}

void KdTreeBuilder::updateCompressEventBBoxAlong(const int &axis)
{
	SplitEvent * eventOffset = &m_event[axis * SplitEvent::NumEventPerDimension];
	
	const float min = m_bbox.getMin(axis);
	const float delta = m_bbox.distance(axis) / SplitEvent::NumBinPerDimension;
	int g, minGrid, maxGrid;
	
	sdb::WorldGrid<GroupCell, unsigned > * grd = m_context->grid();
	
	grd->begin();
	while (!grd->end() ) {
		const BoundingBox & primBox = grd->value()->m_box;
		if(primBox.touch(m_bbox) ) {	
			minGrid = (primBox.getMin(axis) - min) / delta;
		
			if(minGrid < 0) minGrid = 0;
		
			for(g = minGrid; g < SplitEvent::NumEventPerDimension; g++)
				eventOffset[g].updateLeftBox(primBox);
		
			maxGrid = (primBox.getMax(axis) - min) / delta;
		
			if(maxGrid > SplitEvent::NumEventPerDimension) maxGrid = SplitEvent::NumEventPerDimension;

			for(g = maxGrid; g > 0; g--)
				eventOffset[g - 1].updateRightBox(primBox);
		}
		grd->next();
	}
}

void KdTreeBuilder::calculateBins()
{
	const sdb::VectorArray<BoundingBox> & primBoxes = BuildKdTreeContext::GlobalContext->primitiveBoxes();
	const sdb::VectorArray<unsigned> & indices = m_context->indices();
	const unsigned nprim = m_context->getNumPrimitives();
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bbox.distance(axis) < 1e-3f) {
		    m_bins[axis].setFlat();
			continue;
		}
		m_bins[axis].create(SplitEvent::NumBinPerDimension, m_bbox.getMin(axis), m_bbox.getMax(axis));
		for(unsigned i = 0; i < nprim; i++) {
			//std::cout<<" "<<indices[i];
			const BoundingBox * primBox = primBoxes[*indices[i]];
			m_bins[axis].add(primBox->getMin(axis), primBox->getMax(axis));
		}
		
		m_bins[axis].scan();
	}
}

void KdTreeBuilder::calculateSplitEvents()
{
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
	
#if 0
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
#else
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat())
			continue;
			
		updateEventBBoxAlong(axis);
	}
#endif
	
}

void KdTreeBuilder::updateEventBBoxAlong(const int &axis)
{
	SplitEvent * eventOffset = &m_event[axis * SplitEvent::NumEventPerDimension];
	
	const float min = m_bbox.getMin(axis);
	const float delta = m_bbox.distance(axis) / SplitEvent::NumBinPerDimension;
	int g, minGrid, maxGrid;
	const sdb::VectorArray<BoundingBox> & primBoxes = BuildKdTreeContext::GlobalContext->primitiveBoxes();
	const sdb::VectorArray<unsigned> & indices = m_context->indices();
	const unsigned nprim = m_context->getNumPrimitives();
	
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

SplitEvent * KdTreeBuilder::splitAt(int axis, int idx)
{
	return &m_event[axis * SplitEvent::NumEventPerDimension + idx];
}

void KdTreeBuilder::byLowestCost(unsigned & dst)
{
	float lowest = 1e28f;
	
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		for(int i = 1; i < SplitEvent::NumEventPerDimension - 1; i++) {
			SplitEvent * e = splitAt(axis, i);
			if(e->getCost() < lowest && e->hasBothSides()) {
				lowest = e->getCost();
				dst = i + SplitEvent::NumEventPerDimension * axis;
			}
		}
	}
	/// printf("\n lowest cost at %i: %i left %i right %i\n", dst/SplitEvent::NumEventPerDimension,  dst%SplitEvent::NumEventPerDimension, m_event[dst].leftCount(), m_event[dst].rightCount());
}

char KdTreeBuilder::byCutoffEmptySpace(unsigned &dst)
{
	int res = -1;
	float vol, area, emptyVolume = -1.f;
	const int minHead = 2;
	const int maxTail = SplitEvent::NumEventPerDimension - 3;
	const int midSect = SplitEvent::NumEventPerDimension / 2;
	int i, head, tail;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(m_bins[axis].isFlat() ) continue;
		
		area = m_bbox.crossSectionArea(axis);
		
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
	if(res > 0) {
		dst = res;
		/// printf("\n cutoff at %i: %i left %i right %i\n", res/SplitEvent::NumEventPerDimension,  res%SplitEvent::NumEventPerDimension, m_event[res].leftCount(), m_event[res].rightCount());
	}
	return res>0;
}

void KdTreeBuilder::partition(BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx)
{
	
	SplitEvent &e = m_event[m_bestEventIdx];
	
	BoundingBox leftBox, rightBox;
	m_bbox.split(e.getAxis(), e.getPos(), leftBox, rightBox);
	leftCtx.setBBox(leftBox);
	rightCtx.setBBox(rightBox);
	
	if(m_context->isCompressed() )
		partitionCompress(e, leftBox, rightBox, leftCtx, rightCtx);
	else 
		partitionPrims(e, leftBox, rightBox, leftCtx, rightCtx);
	
}

void KdTreeBuilder::partitionCompress(const SplitEvent & e,
					const BoundingBox & leftBox, const BoundingBox & rightBox,
						BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx)
{
	sdb::WorldGrid<GroupCell, unsigned > * grd = m_context->grid();
	
	if(e.leftCount() > 0)
		leftCtx.createGrid(grd->gridSize() );
	if(e.rightCount() > 0)
		rightCtx.createGrid(grd->gridSize() );
	
	int side;
	grd->begin();
	while (!grd->end() ) {
		const BoundingBox & primBox = grd->value()->m_box;
		if(primBox.touch(m_bbox) ) {	
			side = e.side(primBox);
			if(side < 2) {
				//if(primBox.touch(leftBox))
				leftCtx.addCell(grd->key(), grd->value() );
			}
			if(side > 0) {
				//if(primBox.touch(rightBox))
				rightCtx.addCell(grd->key(), grd->value() );
			}
		}
		grd->next();
	}
	 
	if(e.leftCount() > 0) {
		leftCtx.countPrimsInGrid();
#if 0
		const int ncl = leftCtx.numCells();
		if(leftCtx.decompress())
			std::cout<<"\n decomp lft cell "<<ncl<<" n prim "<<leftCtx.getNumPrimitives();
#else
		leftCtx.decompress();
#endif
	}
	if(e.rightCount() > 0) {
		rightCtx.countPrimsInGrid();
#if 0
		const int ncr = rightCtx.numCells();
		if(rightCtx.decompress())
			std::cout<<"\n decomp rgt cell "<<ncr<<" n prim "<<rightCtx.getNumPrimitives();
#else
		rightCtx.decompress();
#endif
	}
	
}

void KdTreeBuilder::partitionPrims(const SplitEvent & e,
					const BoundingBox & leftBox, const BoundingBox & rightBox,
						BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx)
{
	const sdb::VectorArray<BoundingBox> & boxSrc = BuildKdTreeContext::GlobalContext->primitiveBoxes();
	const sdb::VectorArray<unsigned> & indices = m_context->indices();
	
	int side;
	const unsigned nprim = m_context->getNumPrimitives();
	for(unsigned i = 0; i < nprim; i++) {
		
		const BoundingBox * primBox = boxSrc[*indices[i]];
		
		side = e.side(*primBox);
		
		if(side < 2) {
			//if(primBox->touch(leftBox))
			leftCtx.addIndex(*indices[i]);
		}
		if(side > 0) {
			//if(primBox->touch(rightBox))
			rightCtx.addIndex(*indices[i]);
		}		
	}
	
	/// std::cout<<"\n part prim "<<leftCtx.getNumPrimitives()<<"/"<<rightCtx.getNumPrimitives();
}

void KdTreeBuilder::verbose() const
{
	const unsigned nprim = m_context->getNumPrimitives();
	
	printf("unsplit cost %f = 2 * %i box %f\n", 2.f * nprim, nprim, m_bbox.area());
	m_event[m_bestEventIdx].verbose();
	printf("chose split %i: %i\n", m_bestEventIdx/SplitEvent::NumEventPerDimension,  m_bestEventIdx%SplitEvent::NumEventPerDimension);
}

}
