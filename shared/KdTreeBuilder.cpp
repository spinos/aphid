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

KdTreeBuilder::KdTreeBuilder(BuildKdTreeContext &ctx, const PartitionBound &bound) 
{
	m_context = &ctx;
	IndexArray &indices = ctx.indices();
	PrimitiveArray &primitives = ctx.primitives();
	BoundingBox *primBoxes = m_context->m_primitiveBoxes;
	unsigned oldIdx = indices.index();
	
	m_numPrimitive = bound.numPrimitive();
	m_primitives = new PrimitivePtr[m_numPrimitive];
	m_indices = new unsigned[m_numPrimitive];
	m_primitiveClassification = new char[m_numPrimitive];
	m_primitiveBoxes = new BoundingBox[m_numPrimitive];
	indices.setIndex(bound.parentMin);
	primitives.setIndex(bound.parentMin);
	
	for(unsigned i = bound.parentMin; i < bound.parentMax; i++) {
		unsigned primIdx = *indices.asIndex();
		m_indices[i - bound.parentMin] = *indices.asIndex();
		m_primitives[i - bound.parentMin] = primitives.asPrimitive();
		m_primitiveBoxes[i - bound.parentMin] = primBoxes[primIdx];
		indices.next();
		primitives.next();
	}
	
	indices.setIndex(oldIdx);
	primitives.setIndex(oldIdx);
	
	m_bbox = bound.bbox;
	
	calculateBins();
	calculateSplitEvents();
}

KdTreeBuilder::~KdTreeBuilder() 
{
	//printf("builder quit\n");
	delete[] m_event;
	delete[] m_bins;
	delete[] m_primitives;
	delete[] m_indices;
	delete[] m_primitiveClassification;
	delete[] m_primitiveBoxes;
}

void KdTreeBuilder::calculateBins()
{
	m_bins = new MinMaxBins[SplitEvent::Dimension];
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		m_bins[axis].create(33, m_bbox.getMin(axis), m_bbox.getMax(axis));
	
		for(unsigned i = 0; i < m_numPrimitive; i++) {
			BoundingBox &primBox = m_primitiveBoxes[i];
			m_bins[axis].add(primBox.getMin(axis), primBox.getMax(axis));
		}
		
		m_bins[axis].scan();
	}
}

void KdTreeBuilder::calculateSplitEvents()
{
	SplitEvent::NumPrimitive = m_numPrimitive;
	SplitEvent::PrimitiveIndices = m_indices;
	SplitEvent::PrimitiveBoxes = m_primitiveBoxes;
	SplitEvent::ParentBoxArea = m_bbox.area();
	
	const unsigned numEvent = numEvents();
	m_event = new SplitEvent[numEvent];
	int eventIdx = 0;
	unsigned leftNumPrim, rightNumPrim;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		const float min = m_bbox.getMin(axis);
		const float max = m_bbox.getMax(axis);
		const float delta = (max - min) / 33.f;
		for(int i = 1; i < 33; i++) {
			SplitEvent &event = m_event[eventIdx];
			event.setAxis(axis);
			event.setPos(min + delta * i);
			m_bins[axis].get(i - 1, leftNumPrim, rightNumPrim);
			event.setLeftRightNumPrim(leftNumPrim, rightNumPrim);
			eventIdx++;
		}
	}
	
	boost::thread eventThread[NUMEVENTTHREAD];
	
	for(unsigned i = 0; i < NUMEVENTTHREAD; i++) {
		eventThread[i] = boost::thread(&SplitEvent::calculateCost, &m_event[i]);
	}
	
	for(unsigned i = 0; i < NUMEVENTTHREAD; i++) {
		eventThread[i].join();
	}
	
/*
	for(unsigned j = 0; j < m_numPrimitive; j++) {
		unsigned &primIdx = m_indices[j];
		BoundingBox &primBox = primBoxes[primIdx];
		for(unsigned i = 0; i < numEvent; i++) {
			SplitEvent &event = m_event[i];
			event.calculateTightBBoxes(primBox);
		}
		
		//Primitive *prim = m_primitives[i];
		//BaseMesh *mesh = (BaseMesh *)(prim->getGeometry());
		//unsigned triIdx = prim->getComponentIndex();
		
		//mesh->calculateBBox(triIdx);
	}*/
}

void KdTreeBuilder::calculateSides()
{
	BoundingBox *primBoxes = m_context->m_primitiveBoxes;

	SplitEvent &e = m_event[m_bestEventIdx];
	for(unsigned i = 0; i < m_numPrimitive; i++) {
		unsigned &primIdx = m_indices[i];
		BoundingBox &primBox = primBoxes[primIdx];
		//Primitive *prim = m_primitives[i];
		//BaseMesh *mesh = (BaseMesh *)(prim->getGeometry());
		//unsigned triIdx = prim->getComponentIndex();
		//int side = mesh->faceOnSideOf(triIdx, axis, pos);
		m_primitiveClassification[i] = e.side(primBox);
	}
}

const SplitEvent *KdTreeBuilder::bestSplit()
{
	m_bestEventIdx = 0;//axis * 32 + 16;
	float lowest = m_event[0].getCost();
	for(unsigned i = 0; i < numEvents(); i++) {
		//m_event[i].verbose();
		if(m_event[i].getCost() < lowest) {
			lowest = m_event[i].getCost();
			m_bestEventIdx = i;
		}
	}
	cutoffEmptySpace();
	calculateSides();
	return &m_event[m_bestEventIdx];
}

void KdTreeBuilder::cutoffEmptySpace()
{
	IndexLimit emptySpace[3];
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		float preCost = -1.f;
		int eventStart = 32 * axis;
		EmptySpace cutoff;
		for(unsigned i = 0; i < 32; i++) {
			if(m_event[eventStart + i].getCost() != preCost) {
				IndexLimit block;
				block.low = i;
				cutoff.push_back(block);
			}
			else {
				IndexLimit &lastBlock = cutoff.back();
				lastBlock.high = i;
			}
			preCost = m_event[eventStart + i].getCost();
		}
		
		for (std::vector<IndexLimit>::iterator it = cutoff.begin() ; it < cutoff.end(); it++ ) {
			IndexLimit block = *it;
			if(block.high - block.low > 7) {
				if(block.high - block.low > emptySpace[axis].high - emptySpace[axis].low)
					emptySpace[axis] = block;
			}
		}
	}
	
	float emptyArea[3];
	
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(emptySpace[axis].high > emptySpace[axis].low) {
			emptyArea[axis] = m_bbox.distance(axis) * m_bbox.crossSectionArea(axis) * (emptySpace[axis].high - emptySpace[axis].low);
		}
		else
			emptyArea[axis] = -99.f;
	}
	
	float maxEmptySpace = -1.f;
	int maxEmptyAxis = -1;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		if(emptyArea[axis] > maxEmptySpace) {
			maxEmptySpace = emptyArea[axis];
			maxEmptyAxis = axis;
		}
	}
	
	if(maxEmptySpace < 0.f || maxEmptyAxis < 0) return;
	
	//printf("%i: empty %i - %i\n", maxEmptyAxis, emptySpace[maxEmptyAxis].low, emptySpace[maxEmptyAxis].high);
	if(emptySpace[maxEmptyAxis].low == 0)
		m_bestEventIdx = maxEmptyAxis * 32 + emptySpace[maxEmptyAxis].high;
	else
		m_bestEventIdx = maxEmptyAxis * 32 + emptySpace[maxEmptyAxis].low;
}

unsigned KdTreeBuilder::numEvents() const
{
	return 32 * SplitEvent::Dimension;
}

void KdTreeBuilder::partitionLeft(BuildKdTreeContext &ctx, PartitionBound & bound)
{	
	IndexArray &indices = ctx.indices();
	PrimitiveArray &primitives = ctx.primitives();
	
	bound.childMin = indices.index();

	indices.expandBy(m_numPrimitive);
	primitives.expandBy(m_numPrimitive);
	//printf("left side ");
	int count = 0;
	for(unsigned i = bound.parentMin; i < bound.parentMax; i++) {
		int side = m_primitiveClassification[i - bound.parentMin];
		if(side < 2) {
			unsigned idxSrc = m_indices[i - bound.parentMin];
			unsigned *idxDes = indices.asIndex();
			*idxDes = idxSrc;
			
			Primitive *primSrc = m_primitives[i - bound.parentMin];
			Primitive *primDes = primitives.asPrimitive();
			*primDes = *primSrc;
			//primDes->setGeometry((char *)primSrc->getGeometry());
			//primDes->setComponentIndex(primSrc->getComponentIndex());
			//printf(" %i ", *cur);
			indices.next();
			primitives.next();
			count++;
		}
	}
	bound.childMax = indices.index();
	printf("%i to left side\n", count);	
		//printf("left index %i - %i\n", bound.childMin, bound.childMax);
	//printf("ctx partition %i primitives\n", bound.numPrimitive());
	
	//
	
	//for(unsigned i = bound.leftChildMin; i < bound.leftChildMax; i++) {
	//	printf("%i ", *m_indices.asIndex(i));
	//}
	//printf("\n");
	//
	//for(unsigned i = bound.rightChildMin; i < bound.rightChildMax; i++) {
	//	printf("%i ", *m_indices.asIndex(i));
	//}
	//printf("\n");
	
}

void KdTreeBuilder::partitionRight(BuildKdTreeContext &ctx, PartitionBound & bound)
{	
	IndexArray &indices = ctx.indices();
	PrimitiveArray &primitives = ctx.primitives();

	bound.childMin = indices.index();
	//printf("right side ");
	indices.expandBy(m_numPrimitive);
	primitives.expandBy(m_numPrimitive);
	int count = 0;
	for(unsigned i = bound.parentMin; i < bound.parentMax; i++) {
		int side = m_primitiveClassification[i - bound.parentMin];
		if(side > 0) {
			unsigned idxSrc = m_indices[i - bound.parentMin];
			unsigned *idxDes = indices.asIndex();
			*idxDes = idxSrc;
			
			Primitive *primSrc = m_primitives[i - bound.parentMin];
			Primitive *primDes = primitives.asPrimitive();
			*primDes = *primSrc;
			//primDes->setGeometry((char *)primSrc->getGeometry());
			//primDes->setComponentIndex(primSrc->getComponentIndex());
			//printf(" %i ", *cur);
			indices.next();
			primitives.next();
			count++;
		}
	}
	bound.childMax = indices.index();
	printf("%i to right side\n", count);
}

