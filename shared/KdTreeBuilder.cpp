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

KdTreeBuilder::KdTreeBuilder() {}
KdTreeBuilder::~KdTreeBuilder() {}

void KdTreeBuilder::setContext(BuildKdTreeContext &ctx) 
{
	m_context = &ctx;
	m_bbox = ctx.getBBox();
	
	initBins(m_bbox);
	
	if(m_context->isCompressed() ) {
		calculateCompressBins(m_context->grid(), m_bbox);
		initEvents(m_bbox);
		calculateCompressSplitEvents(m_context->grid(), m_bbox);
	}
	else {
		calculateBins(m_context->getNumPrimitives(),
				m_context->indices(),
				BuildKdTreeContext::GlobalContext->primitiveBoxes() );
		initEvents(m_bbox);
		calculateSplitEvents(m_bbox,
				m_context->getNumPrimitives(),
				m_context->indices(),
				BuildKdTreeContext::GlobalContext->primitiveBoxes() );
	}
	
	calculateCosts(m_bbox);
}

const SplitEvent *KdTreeBuilder::bestSplit()
{
	splitAtLowestCost(m_bbox);
	return &m_event[m_bestEventIdx];
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
