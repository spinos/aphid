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
	IndexArray &indices = ctx.indices();
	PrimitiveArray &primitives = ctx.primitives();
	
	unsigned oldIdx = indices.index();
	
	unsigned numPrim = bound.numPrimitive();
	m_primitives = new PrimitivePtr[numPrim];
	
	indices.setIndex(bound.parentMin);
	primitives.setIndex(bound.parentMin);
	for(unsigned i = bound.parentMin; i < bound.parentMax; i++) {
		//unsigned idx = *indices.asIndex();
		m_primitives[i - bound.parentMin] = primitives.asPrimitive();
		indices.next();
		primitives.next();
	}
	
	indices.setIndex(oldIdx);
	primitives.setIndex(oldIdx);
	
	m_bbox = bound.bbox;
}

KdTreeBuilder::~KdTreeBuilder() 
{
	//printf("builder quit\n");
	delete[] m_event;
	delete[] m_primitives;
	delete[] m_primitiveClassification;
}

void KdTreeBuilder::calculateSplitEvents(const PartitionBound &bound)
{
	m_numPrimitive = bound.numPrimitive();
	m_event = new SplitEvent[numEvents()];
	m_primitiveClassification = new char[m_numPrimitive];
	int eventIdx = 0;
	for(int axis = 0; axis < SplitEvent::Dimension; axis++) {
		float min = bound.bbox.getMin(axis);
		float max = bound.bbox.getMax(axis);
		float delta = (max - min) / 33.f;
		for(int i = 1; i < 33; i++) {
			SplitEvent &event = m_event[eventIdx];
			event.setAxis(axis);
			event.setPos(min + delta * i);
			eventIdx++;
		}
	}
	
}

void KdTreeBuilder::calculateSides(const unsigned &eventIdx)
{
	SplitEvent &e = m_event[eventIdx];
	int axis = e.getAxis();
	float pos = e.getPos();
	for(unsigned i = 0; i < m_numPrimitive; i++) {
		Primitive *prim = m_primitives[i];
		BaseMesh *mesh = (BaseMesh *)(prim->getGeometry());
		unsigned triIdx = prim->getComponentIndex();
		int side = mesh->faceOnSideOf(triIdx, axis, pos);
		m_primitiveClassification[i] = side;
	}
}

const SplitEvent *KdTreeBuilder::bestSplit()
{
	int axis = m_bbox.getLongestAxis();
	int iEvent = axis * 32 + 15;
	calculateSides(iEvent);
	return &m_event[iEvent];
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
			//unsigned idx = *m_indices.asIndex(i);
			//unsigned *cur = m_indices.asIndex();
			//*cur = idx;
			
			Primitive *primSrc = m_primitives[i - bound.parentMin];
			Primitive *primDes = primitives.asPrimitive();
			primDes->setGeometry((char *)primSrc->getGeometry());
			primDes->setComponentIndex(primSrc->getComponentIndex());
			//printf(" %i ", *cur);
			indices.next();
			primitives.next();
			count++;
		}
	}
	bound.childMax = indices.index();
	//printf("%i to left side\n", count);	
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
			//unsigned idx = *m_indices.asIndex(i);
			//unsigned *cur = m_indices.asIndex();
			//*cur = idx;
			
			Primitive *primSrc = m_primitives[i - bound.parentMin];
			Primitive *primDes = primitives.asPrimitive();
			primDes->setGeometry((char *)primSrc->getGeometry());
			primDes->setComponentIndex(primSrc->getComponentIndex());
			//printf(" %i ", *cur);
			indices.next();
			primitives.next();
			count++;
		}
	}
	bound.childMax = indices.index();
	//printf("%i to right side\n", count);
		//printf("right index %i - %i\n", bound.childMin, bound.childMax);
	
	
}

