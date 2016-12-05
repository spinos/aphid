/*
 *  KdTreeBuilder.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <kd/KdTreeBuilder.h>
#include <boost/thread.hpp>  

namespace aphid {

int KdTreeBuilder::MaxLeafPrimThreashold = 256;
int KdTreeBuilder::MaxBuildLevel = 32;
	
KdTreeBuilder::KdTreeBuilder() {}
KdTreeBuilder::~KdTreeBuilder() {}

void KdTreeBuilder::setContext(BuildKdTreeContext &ctx) 
{
	m_context = &ctx;
	m_bbox = ctx.getBBox();
	
	if(m_context->isCompressed() )
		calcSoftBin(m_context->grid(), m_bbox);
	else
		calcSoftBin(m_context->numPrims(),
				m_context->indices(),
				BuildKdTreeContext::GlobalContext->primitiveBoxes(),
				m_bbox );
				
	initEvents(m_bbox);

	if(m_context->isCompressed() )
		calcEvent(m_context->grid(), m_bbox);
	else
		calcEvent(m_bbox,
				m_context->numPrims(),
				m_context->indices(),
				BuildKdTreeContext::GlobalContext->primitiveBoxes() );
	
	
	calculateCosts(m_bbox);
}

const SplitEvent *KdTreeBuilder::bestSplit()
{
	splitAtLowestCost(m_bbox);
	return split(m_bestEventIdx);
}

void KdTreeBuilder::partition(BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx)
{
	
	const SplitEvent *e = split(m_bestEventIdx);
	
	BoundingBox leftBox, rightBox;
	m_bbox.split(e->getAxis(), e->getPos(), leftBox, rightBox);
	leftCtx.setBBox(leftBox);
	rightCtx.setBBox(rightBox);
	
	if(m_context->isCompressed() )
		partitionCompress(*e, leftBox, rightBox, leftCtx, rightCtx);
	else 
		partitionPrims(*e, leftBox, rightBox, leftCtx, rightCtx);
	
}

void KdTreeBuilder::partitionCompress(const SplitEvent & e,
					const BoundingBox & leftBox, const BoundingBox & rightBox,
						BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx)
{
	sdb::WorldGrid<sdb::GroupCell, unsigned > * grd = m_context->grid();
	
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
	 
	if(e.leftCount() > 0)
		leftCtx.decompressPrimitives();
	
	if(e.rightCount() > 0)
		rightCtx.decompressPrimitives();
	
}

void KdTreeBuilder::partitionPrims(const SplitEvent & e,
					const BoundingBox & leftBox, const BoundingBox & rightBox,
						BuildKdTreeContext &leftCtx, BuildKdTreeContext &rightCtx)
{
	const sdb::VectorArray<BoundingBox> & boxSrc = BuildKdTreeContext::GlobalContext->primitiveBoxes();
	const sdb::VectorArray<unsigned> & indices = m_context->indices();
	
	int side;
	const int nprim = m_context->numPrims();
	for(int i = 0; i < nprim; i++) {
		
		const BoundingBox * primBox = boxSrc[*indices[i]];
		
		side = e.side(*primBox);
		
		if(side < 2) {
			//if(primBox->touch(leftBox))
			leftCtx.addPrimitive(*indices[i]);
		}
		if(side > 0) {
			//if(primBox->touch(rightBox))
			rightCtx.addPrimitive(*indices[i]);
		}		
	}
	
	/// std::cout<<"\n part prim "<<leftCtx.getNumPrimitives()<<"/"<<rightCtx.getNumPrimitives();
}

}
