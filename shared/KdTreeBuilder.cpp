/*
 *  KdTreeBuilder.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "KdTreeBuilder.h"

KdTreeBuilder::KdTreeBuilder(BuildKdTreeContext &ctx, const PartitionBound &bound) 
{
	m_context = &ctx;
	IndexArray &indices = ctx.indices();
	PrimitiveArray &primitives = ctx.primitives();
	
	unsigned oldIdx = indices.index();
	
	m_numPrimitive = bound.numPrimitive();
	m_primitives = new PrimitivePtr[m_numPrimitive];
	m_indices = new unsigned[m_numPrimitive];
	m_primitiveClassification = new char[m_numPrimitive];
	indices.setIndex(bound.parentMin);
	primitives.setIndex(bound.parentMin);
	for(unsigned i = bound.parentMin; i < bound.parentMax; i++) {
		m_indices[i - bound.parentMin] = *indices.asIndex();
		m_primitives[i - bound.parentMin] = primitives.asPrimitive();
		indices.next();
		primitives.next();
	}
	
	indices.setIndex(oldIdx);
	primitives.setIndex(oldIdx);
	
	calculateSplitEvents(bound);
}

KdTreeBuilder::~KdTreeBuilder() 
{
	//printf("builder quit\n");
	delete[] m_event;
	delete[] m_primitives;
	delete[] m_indices;
	delete[] m_primitiveClassification;
}

void KdTreeBuilder::calculateSplitEvents(const PartitionBound &bound)
{
	m_bbox = bound.bbox;
	BoundingBox *primBoxes = m_context->m_primitiveBoxes;
	
	const unsigned numEvent = numEvents();
	m_event = new SplitEvent[numEvent];
	int eventIdx = 0;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		const float min = bound.bbox.getMin(axis);
		const float max = bound.bbox.getMax(axis);
		const float delta = (max - min) / 33.f;
		for(int i = 1; i < 33; i++) {
			SplitEvent &event = m_event[eventIdx];
			event.setAxis(axis);
			event.setPos(min + delta * i);
			eventIdx++;
		}
	}

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
	}
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
	int axis = m_bbox.getLongestAxis();
	m_bestEventIdx = axis * 32 + 15;
	calculateSides();
	return &m_event[m_bestEventIdx];
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

