/*
 *  KdEngine.h
 *  testntree
 *
 *  Created by jian zhang on 11/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include "KdNTree.h"
#include "KdBuilder.h"
#include <IntersectionContext.h>
#include <SelectionContext.h>
#include <Geometry.h>

namespace aphid {

class KdEngine {

    int m_numRopeTraversed;
    
public:
	template<typename T, typename Ts>
	void buildSource(sdb::VectorArray<T> * dst,
					BoundingBox & box,
					const std::vector<Ts *> & src);
	
	template<typename T, typename Tn, int NLevel>
	void buildTree(KdNTree<T, Tn > * tree, 
					sdb::VectorArray<T> * source, const BoundingBox & box,
					const TreeProperty::BuildProfile * prof);
	
	template<typename T, typename Tn>
	void printTree(KdNTree<T, Tn > * tree);
	
	template<typename T, typename Tn>
	bool intersect(KdNTree<T, Tn > * tree, 
				IntersectionContext * ctx);
	
	template<typename T, typename Tn>
	void select(KdNTree<T, Tn > * tree, 
				SphereSelectionContext * ctx);
				
	template<typename T, typename Tn>
	void closestToPoint(KdNTree<T, Tn > * tree, 
				Geometry::ClosestToPointTestResult * ctx);
				
	template<typename T, typename Tn>
	void intersectBox(KdNTree<T, Tn > * tree, 
				BoxIntersectContext * ctx);

/// domain of element much smaller than bx
	template<typename T, typename Tn>
	bool broadphase(KdNTree<T, Tn > * tree,
				const BoundingBox & bx);
	
protected:

private:
	template<typename T, typename Tn>
	void printBranch(KdNTree<T, Tn > * tree, int idx);
	
	template<typename T, typename Tn>
	void leafSelect(KdNTree<T, Tn > * tree, 
					SphereSelectionContext * ctx,
					KdTreeNode * r);
	
	template<typename T, typename Tn>
	void innerSelect(KdNTree<T, Tn > * tree, 
					SphereSelectionContext * ctx,
					int branchIdx,
					int nodeIdx,
					const BoundingBox & b);
	
/// 0 or 1
	template <typename T>
	int firstVisit(IntersectionContext * ctx, 
					const KdTreeNode * n);
	
	template <typename T, typename Tn>
	int visitLeaf(IntersectionContext * ctx, 
					int & branchIdx,
					int & nodeIdx,
					const KdTreeNode * r);
					
	template <typename T, typename Tn>
	int hitPrimitive(KdNTree<T, Tn > * tree, 
							IntersectionContext * ctx, 
							const KdTreeNode * r);
	
	template <typename T, typename Tn>
	bool climbRope(KdNTree<T, Tn > * tree, 
							IntersectionContext * ctx, 
									int & branchIdx,
									int & nodeIdx,
									const KdTreeNode * r);
									
	template<typename T, typename Tn>
	void leafClosestToPoint(KdNTree<T, Tn > * tree, 
								Geometry::ClosestToPointTestResult * result,
								KdTreeNode *node, const BoundingBox &box);
								
	template<typename T, typename Tn>
	void innerClosestToPoint(KdNTree<T, Tn > * tree, 
				Geometry::ClosestToPointTestResult * ctx,
				int branchIdx,
				int nodeIdx,
				const BoundingBox & b);
				
	template<typename T, typename Tn>
	void leafIntersectBox(KdNTree<T, Tn > * tree,
				BoxIntersectContext * ctx,
				KdTreeNode * r);
		
	template <typename T, typename Tn>
	void innerIntersectBox(KdNTree<T, Tn > * tree,
				BoxIntersectContext * ctx,
				int branchIdx,
				int nodeIdx,
				const BoundingBox & b);
				
	template<typename T, typename Tn>
	bool leafBroadphase(KdNTree<T, Tn > * tree,
				const BoundingBox * b,
				KdTreeNode * r);
				
	template <typename T, typename Tn>
	bool innerBroadphase(KdNTree<T, Tn > * tree,
				const BoundingBox * b,
				int branchIdx,
				int nodeIdx,
				const BoundingBox & innerBx);
				
};

template<typename T, typename Ts>
void KdEngine::buildSource(sdb::VectorArray<T> * dst,
					BoundingBox & box,
					const std::vector<Ts *> & src)
{
	box.reset();
	T acomp;
	unsigned a, b, c;
	typename std::vector<Ts *>::const_iterator it = src.begin();
    for(int i=0;it!=src.end();++it,++i) {
/// ind 0 to geom
		acomp.setInd(i, 0);
		Ts * w = *it;
		const Vector3F * p = w->points();
		unsigned n = w->numComponents();
		for(unsigned j=0; j<n; ++j) {
			unsigned * v = w->triangleIndices(j);
			a = v[0];
			b = v[1];
			c = v[2];
			box.expandBy(p[a], 1e-4f);
			acomp.setP(p[a], 0);
			box.expandBy(p[b], 1e-4f);
			acomp.setP(p[b], 1);
			box.expandBy(p[c], 1e-4f);
			acomp.setP(p[c], 2);
/// ind 1 to component
			acomp.setInd(j, 1);
			dst->insert(acomp);
		}
	}
    box.round();
}

template<typename T, typename Tn, int NLevel>
void KdEngine::buildTree(KdNTree<T, Tn > * tree, 
							sdb::VectorArray<T> * source, const BoundingBox & box,
							const TreeProperty::BuildProfile * prof)
{
	tree->init(source, box);
    
    std::cout<<"\n kdengine begin building "<<T::GetTypeStr()<<" tree "
            <<"\n bbx "<<box
			<<"\n max n prims per leaf "<<prof->_maxLeafPrims
			<<"\n max build level "<<prof->_maxLevel;
    
    KdNBuilder<NLevel, T, Tn > bud;
	bud.SetNumPrimsInLeaf(prof->_maxLeafPrims);
	bud.MaxTreeletLevel = prof->_maxLevel;
	
/// first split
	SahSplit<T> splt(source);
	splt.setBBox(box);
	splt.initIndicesAndBoxes(source->size() );
    
	SahSplit<T>::GlobalSplitContext = &splt;
	
	bud.build(&splt, tree);
	if(prof->_doTightBox) tree->storeTightBox();
	tree->verbose();
}

template<typename T, typename Tn>
void KdEngine::printTree(KdNTree<T, Tn > * tree)
{
	Tn * tn = tree->root();
	std::cout<<"\n root";
	tn->verbose();
	int i=0;
	KdTreeNode * child = tn->node(0);
	if(child->isLeaf() ) {}
	else {
		printBranch<T, Tn>(tree, tn->internalOffset(0) );
	}
}

template<typename T, typename Tn>
void KdEngine::printBranch(KdNTree<T, Tn > * tree, int idx)
{
	Tn * tn = tree->branches()[idx];
	std::cout<<"\n branch["<<idx<<"]";
	tn->verbose();
	int i=14;
	for(;i<Tn::NumNodes;++i) {
		KdTreeNode * child = tn->node(i);
		if(child->isLeaf() ) {}
		else {
			if(tn->internalOffset(i) > 0) printBranch(tree, idx + tn->internalOffset(i) );
		}
	}
}

template<typename T, typename Tn>
bool KdEngine::intersect(KdNTree<T, Tn > * tree, 
				IntersectionContext * ctx)
{
	if(tree->isEmpty()) return 0;
	
	const BoundingBox & b = tree->getBBox();
	if(!b.intersect(ctx->m_ray)) return 0;
	
	std::cout<<"\n KdEngine::intersect"<<b;
	m_numRopeTraversed = 0;
    
	ctx->setBBox(b);
	
	const KdTreeNode * r = tree->root()->node(0);
	if(r->isLeaf() ) {
		ctx->m_leafIdx = r->getPrimStart();
		hitPrimitive(tree, ctx, r);
		return ctx->m_success;
	}
	
	int branchIdx = tree->root()->internalOffset(0);
	int preBranchIdx = branchIdx;
	Tn * currentBranch = tree->branches()[branchIdx];
	int nodeIdx = firstVisit<T>(ctx, r);
	KdTreeNode * kn = currentBranch->node(nodeIdx);
	int stat;
	bool hasNext = true;
	while (hasNext) {
		stat = visitLeaf<T, Tn>(ctx, branchIdx, nodeIdx, 
							kn);
							
		if(preBranchIdx != branchIdx) {
			currentBranch = tree->branches()[branchIdx];
			preBranchIdx = branchIdx;
		}
		
		kn = currentBranch->node(nodeIdx);
		
		if(stat > 0 ) {
			if(hitPrimitive(tree, ctx, kn ) ) hasNext = false;
			else stat = 0;
		}
		if(stat==0) {
			hasNext = climbRope(tree, ctx, branchIdx, nodeIdx, 
							kn );
							
			if(preBranchIdx != branchIdx) {
				currentBranch = tree->branches()[branchIdx];
				preBranchIdx = branchIdx;
			}
			
			kn = currentBranch->node(nodeIdx);
		}
	}
	return ctx->m_success;
}

template <typename T>
int KdEngine::firstVisit(IntersectionContext * ctx, 
								const KdTreeNode * n)
{
#if 1
    const BoundingBox & b = ctx->getBBox();
	b.intersect(ctx->m_ray, &ctx->m_tmin, &ctx->m_tmax);
	
	Vector3F enterP = ctx->m_ray.travel(ctx->m_tmin);
	const int axis = n->getAxis();
	const float splitPos = n->getSplitPos();
	int above = enterP.comp(axis) >= splitPos;
	BoundingBox childBox;
	
	if(above) 
		b.splitRight(axis, splitPos, childBox);
	else
		b.splitLeft(axis, splitPos, childBox);
		
	ctx->setBBox(childBox);
		
	return above; 
#else
	const int axis = n->getAxis();
	const float splitPos = n->getSplitPos();
	
	const Ray & ray = ctx->m_ray;
	
	const float o = ray.m_origin.comp(axis);
	const float d = ray.m_dir.comp(axis);
	
	const BoundingBox & b = ctx->getBBox();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	// std::cout<<"\n split "<<axis<<" "<<splitPos;
	
	bool above = o >= splitPos;
/// parallel to split
	if(Absolute<float>(d) < 1e-5f) {
		if(above) ctx->setBBox(rgtBox);
		else ctx->setBBox(lftBox);
		
		return above;
	}
	
/// inside
	if(lftBox.isPointInside(ray.m_origin) ) {
		ctx->setBBox(lftBox);
		return 0;
	}
	
	if(rgtBox.isPointInside(ray.m_origin) ) {
		ctx->setBBox(rgtBox);
		return 1;
	}
	
/// one side
	float t = (splitPos - o) / d;
	if(t< 0.f) {
		if(above) ctx->setBBox(rgtBox);
		else ctx->setBBox(lftBox);
		return above;
	}
	
/// near one
	Vector3F hitP = ray.travel(t);
	if(b.isPointInside(hitP) ) {
		if(above) ctx->setBBox(rgtBox);
		else ctx->setBBox(lftBox);
		return above;
	}
	
	BoundingBox p2h;
	p2h.expandBy(ray.m_origin);
	p2h.expandBy(hitP);
/// swap side
	if(!p2h.touch(b) ) above = !above;
	
	if(above) ctx->setBBox(rgtBox);
	else ctx->setBBox(lftBox);
		
	return above;
#endif
}

template <typename T, typename Tn>
int KdEngine::visitLeaf(IntersectionContext * ctx, 
									int & branchIdx,
									int & nodeIdx,
									const KdTreeNode * r)
{
//	std::cout<<"\n node "<<nodeIdx;
				
	if(r->isLeaf() ) {
//		std::cout<<"\n hit leaf "<<r->getPrimStart();
		ctx->m_leafIdx = r->getPrimStart();
		if(r->getNumPrims() < 1) {
			return 0;
		}
		return 1;
	}
	
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		// std::cout<<"\n inner offset "<<offset;
		nodeIdx += offset + firstVisit<T>(ctx, r);
	}
	else {
		branchIdx += offset & Tn::TreeletOffsetMaskTau;
//		std::cout<<"\n branch "<<branchIdx;
		nodeIdx = firstVisit<T>(ctx, r);
	}
	
	return -1;
}

template <typename T, typename Tn>
int KdEngine::hitPrimitive(KdNTree<T, Tn > * tree, 
							IntersectionContext * ctx, 
							const KdTreeNode * r)
{
	int start, len;
	tree->leafPrimStartLength(start, len, r->getPrimStart() );
//	std::cout<<"\n n prim "<<len;
	int nhit = 0;
	int i = 0;
	for(;i<len;++i) {
		const T * c = tree->getSource(start + i);
		if(c->intersect(ctx->m_ray, &ctx->m_tmin, &ctx->m_tmax) ) {
			ctx->m_hitP = ctx->m_ray.travel(ctx->m_tmin);
			ctx->m_hitN = c->calculateNormal();
			ctx->m_ray.m_tmax = ctx->m_tmin;
			ctx->m_success = 1;
/// idx of source
			ctx->m_componentIdx = tree->primIndirectionAt(start + i);
			nhit++;
		}
	}
//	if(nhit<1) std::cout<<" no hit ";
	return nhit;
}

template <typename T, typename Tn>
bool KdEngine::climbRope(KdNTree<T, Tn > * tree, 
							IntersectionContext * ctx, 
									int & branchIdx,
									int & nodeIdx,
									const KdTreeNode * r)
{
    m_numRopeTraversed++;
/// limit rope traverse
    if(m_numRopeTraversed > 99)
        return false;
    
    const int oldBranch = branchIdx;
    const int oldNode = nodeIdx;
    
	const BoundingBox & b = ctx->getBBox();
	float t0, t1;
	b.intersect(ctx->m_ray, &t0, &t1);
	Vector3F hit1 = ctx->m_ray.travel(t1 + ctx->m_tdelta);
/// end inside 
	if(b.isPointInside(hit1) ) {
		std::cout<<"\n KdEngine::climbRope "<<hit1<<" out of "<<b;
		return false;
	}
		
	int side = b.pointOnSide(hit1);
///	std::cout<<"\n rope side "<<side;
	
/// leaf ind actually 
	int iLeaf = r->getPrimStart();
	int iRope = tree->leafRopeInd(iLeaf, side);
	
	if(iRope < 1) {
//		std::cout<<" no rope";
		return false;
	}
	
//    std::cout<<" rope["<<iRope<<"]";
	const BoundingBox * rp = tree->ropes()[ iRope ];
	BoxNeighbors::DecodeTreeletNodeHash(rp->m_padding1, KdNode4::BranchingFactor, 
					branchIdx, nodeIdx);
    
    std::cout<<"\n rope branch "<<oldBranch<<":"<<oldNode
    <<" to "<<branchIdx<<":"<<nodeIdx;
            
    std::cout.flush();
	ctx->setBBox(*rp);
	return true;
}

template<typename T, typename Tn>
void KdEngine::select(KdNTree<T, Tn > * tree, 
						SphereSelectionContext * ctx)
{
	if(tree->isEmpty() ) return;
	
	const BoundingBox b = tree->getBBox();
	if(!b.intersect(*ctx)) return;

	KdTreeNode * r = tree->root()->node(0);
	if(r->isLeaf() ) {
		leafSelect(tree, ctx, r);
		return;
	}
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
		
	int branchIdx = tree->root()->internalOffset(0);
	innerSelect(tree, ctx, branchIdx, 0, lftBox);
	innerSelect(tree, ctx, branchIdx, 1, rgtBox);
	
}

template<typename T, typename Tn>
void KdEngine::leafSelect(KdNTree<T, Tn > * tree, 
				SphereSelectionContext * ctx,
				KdTreeNode * r)
{
	if(r->getNumPrims() < 1) return;
	int start, len;
	tree->leafPrimStartLength(start, len, r->getPrimStart() );
	gjk::Sphere sp = ctx->sphere();
	int i = 0;
	for(;i<len;++i) {
		const T * c = tree->getSource(start + i );
		if(c->calculateBBox().intersect(*ctx) ) {
			if(ctx->isExact() ) {
				if(c-> template exactIntersect<gjk::Sphere >(sp ) )
					ctx->addPrim(tree->primIndirectionAt(start + i) );
			}
			else
				ctx->addPrim(tree->primIndirectionAt(start + i) );
		}
	}
}

template<typename T, typename Tn>
void KdEngine::innerSelect(KdNTree<T, Tn > * tree, 
				SphereSelectionContext * ctx,
				int branchIdx,
				int nodeIdx,
				const BoundingBox & b)
{
	Tn * currentBranch = tree->branches()[branchIdx];
	KdTreeNode * r = currentBranch->node(nodeIdx);
	if(r->isLeaf() ) {
		leafSelect(tree, ctx, r);
		return;
	}
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		if(ctx->getMin(axis) < splitPos ) {
			innerSelect(tree, ctx, 
							branchIdx,
							nodeIdx + offset,
							lftBox);
		}
		
		if(ctx->getMax(axis) > splitPos ) {
			innerSelect(tree, ctx, 
							branchIdx,
							nodeIdx + offset + 1,
							rgtBox);
		}
	}
	else {
		if(ctx->getMin(axis) < splitPos ) {
			innerSelect(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
		}
		
		if(ctx->getMax(axis) > splitPos ) {
			innerSelect(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
		}
	}
}

template<typename T, typename Tn>
void KdEngine::closestToPoint(KdNTree<T, Tn > * tree, 
				Geometry::ClosestToPointTestResult * ctx)
{
	if(ctx->closeEnough() ) return;
	if(tree->isEmpty() ) return;
	const BoundingBox b = tree->getBBox();
	KdTreeNode * r = tree->root()->node(0);
	if(r->isLeaf() ) {
		leafClosestToPoint(tree, ctx, r, b);
		return;
	}
	int branchIdx = tree->root()->internalOffset(0);
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox leftBox, rightBox;
	b.split(axis, splitPos, leftBox, rightBox);
	
	const float cp = ctx->_toPoint.comp(axis) - splitPos;
	if(cp < 0.f) {
		innerClosestToPoint(tree, ctx, branchIdx, 0, leftBox);
		if(ctx->closeEnough() ) return;
		if( -cp < ctx->_distance) 
			innerClosestToPoint(tree, ctx, branchIdx, 1, rightBox);
	}
	else {
		innerClosestToPoint(tree, ctx, branchIdx, 1, rightBox);
		if(ctx->closeEnough() ) return;
		if(cp < ctx->_distance)
			innerClosestToPoint(tree, ctx, branchIdx, 0, leftBox);
	}
	
}
	
template<typename T, typename Tn>
void KdEngine::innerClosestToPoint(KdNTree<T, Tn > * tree, 
				Geometry::ClosestToPointTestResult * ctx,
				int branchIdx,
				int nodeIdx,
				const BoundingBox & b)
{
	if(!ctx->closeTo(b) )
		return;
		
	Tn * currentBranch = tree->branches()[branchIdx];
	KdTreeNode * r = currentBranch->node(nodeIdx);
	if(r->isLeaf() ) {
		leafClosestToPoint(tree, ctx, r, b);
		return;
	}
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	const float cp = ctx->_toPoint.comp(axis) - splitPos;
	
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		if(cp < 0.f ) {
			innerClosestToPoint(tree, ctx, 
							branchIdx,
							nodeIdx + offset,
							lftBox);
			
			if(ctx->closeEnough() ) return;
			
			if( -cp < ctx->_distance) 
				innerClosestToPoint(tree, ctx, 
							branchIdx, 
							nodeIdx + offset + 1, 
							rgtBox);
		}
		else {
			innerClosestToPoint(tree, ctx, 
							branchIdx,
							nodeIdx + offset + 1,
							rgtBox);
							
			if(ctx->closeEnough() ) return;
			
			innerClosestToPoint(tree, ctx, 
							branchIdx,
							nodeIdx + offset,
							lftBox);
		}
	}
	else {
		if(cp < 0.f ) {
			innerClosestToPoint(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
							
			if(ctx->closeEnough() ) return;
			
			innerClosestToPoint(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
		}
		else {
			innerClosestToPoint(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
							
			if(ctx->closeEnough() ) return;
			
			innerClosestToPoint(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
		}
	}
}

template<typename T, typename Tn>
void KdEngine::leafClosestToPoint(KdNTree<T, Tn > * tree, 
								Geometry::ClosestToPointTestResult * result,
								KdTreeNode *node, const BoundingBox &box)
{
	if(node->getNumPrims() < 1) return;
	int start, len;
	tree->leafPrimStartLength(start, len, node->getPrimStart() );
	int i = 0;
	for(;i<len;++i) {
		const T * c = tree->getSource(start + i );
		c-> template closestToPoint<Geometry::ClosestToPointTestResult>(result);
	}
	
}


template<typename T, typename Tn>
void KdEngine::intersectBox(KdNTree<T, Tn > * tree, 
				BoxIntersectContext * ctx)
{
	if(tree->isEmpty() ) return;
	KdTreeNode * r = tree->root()->node(0);
	if(r->isLeaf() ) {
		leafIntersectBox(tree, ctx, r);
		return;
	}
	const BoundingBox b = tree->getBBox();
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	int branchIdx = tree->root()->internalOffset(0);
	if(ctx->getMin(axis) < splitPos) 
		innerIntersectBox(tree, ctx, branchIdx, 0, lftBox);
	if(ctx->isFull() ) return;
	
	if(ctx->getMax(axis) > splitPos) 
		innerIntersectBox(tree, ctx, branchIdx, 1, rgtBox);
}

template <typename T, typename Tn>
void KdEngine::leafIntersectBox(KdNTree<T, Tn > * tree,
							BoxIntersectContext * ctx,
							KdTreeNode * r)
{
	if(r->getNumPrims() < 1) return;
	int start, len;
	tree->leafPrimStartLength(start, len, r->getPrimStart() );
	int i = 0;
	for(;i<len;++i) {
		const T * c = tree->getSource(start + i );
        bool hit = c->calculateBBox().intersect(*ctx);
		if(hit) {
			if(ctx->isExact() ) {
                if(!c->calculateBBox().inside(*ctx) )
                    hit = c-> template exactIntersect<BoxIntersectContext >(*ctx);
            }
		}
        
		if(hit) {
			ctx->addPrim(tree->primIndirectionAt(start + i) );
			if(ctx->isFull() ) return;
		}
	}
}

template <typename T, typename Tn>
void KdEngine::innerIntersectBox(KdNTree<T, Tn > * tree,
							BoxIntersectContext * ctx,
							int branchIdx,
							int nodeIdx,
							const BoundingBox & b)
{
	Tn * currentBranch = tree->branches()[branchIdx];
	KdTreeNode * r = currentBranch->node(nodeIdx);
	if(r->isLeaf() ) {
		leafIntersectBox(tree, ctx, r);
		return;
	}
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		if(ctx->getMin(axis) < splitPos ) {
			innerIntersectBox(tree, ctx, 
							branchIdx,
							nodeIdx + offset,
							lftBox);
			if(ctx->isFull() ) return;
		}
		
		if(ctx->getMax(axis) > splitPos ) {
			innerIntersectBox(tree, ctx, 
							branchIdx,
							nodeIdx + offset + 1,
							rgtBox);
			if(ctx->isFull() ) return;
		}
	}
	else {
		if(ctx->getMin(axis) < splitPos ) {
			innerIntersectBox(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
			if(ctx->isFull() ) return;
		}
		
		if(ctx->getMax(axis) > splitPos ) {
			innerIntersectBox(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
			if(ctx->isFull() ) return;
		}
	}
}

template<typename T, typename Tn>
bool KdEngine::leafBroadphase(KdNTree<T, Tn > * tree,
				const BoundingBox * b,
				KdTreeNode * r)
{
	if(r->getNumPrims() < 1) return false;
	int start, len;
	tree->leafPrimStartLength(start, len, r->getPrimStart() );
	int i = 0;
	for(;i<len;++i) {
		const T * c = tree->getSource(start + i );
        if(b->intersect(c->calculateBBox() ) )
			return true;
			
	}
	return false;
}

template <typename T, typename Tn>
bool KdEngine::innerBroadphase(KdNTree<T, Tn > * tree,
				const BoundingBox * b,
				int branchIdx,
				int nodeIdx,
				const BoundingBox & innerBx)
{
	Tn * currentBranch = tree->branches()[branchIdx];
	KdTreeNode * r = currentBranch->node(nodeIdx);
	if(r->isLeaf() ) {
		return leafBroadphase(tree, b, r);
	}
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	innerBx.split(axis, splitPos, lftBox, rgtBox);
	
	bool stat = false;
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		if(b->getMin(axis) < splitPos ) {
			stat = innerBroadphase(tree, b, 
							branchIdx,
							nodeIdx + offset,
							lftBox);
			if(stat ) return stat;
		}
		
		if(b->getMax(axis) > splitPos ) {
			stat = innerBroadphase(tree, b, 
							branchIdx,
							nodeIdx + offset + 1,
							rgtBox);
			if(stat ) return stat;
		}
	}
	else {
		if(b->getMin(axis) < splitPos ) {
			stat = innerBroadphase(tree, b, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
			if(stat ) return stat;
		}
		
		if(b->getMax(axis) > splitPos ) {
			stat = innerBroadphase(tree, b, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
			if(stat ) return stat;
		}
	}
	return stat;
}

template<typename T, typename Tn>
bool KdEngine::broadphase(KdNTree<T, Tn > * tree,
				const BoundingBox & bx)
{
	KdTreeNode * r = tree->root()->node(0);
	if(r->isLeaf() )
		return leafBroadphase(tree, &bx, r);
	
	const BoundingBox b = tree->getBBox();
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	
	bool stat = false;
	int branchIdx = tree->root()->internalOffset(0);
	if(bx.getMin(axis) < splitPos) 
		stat = innerBroadphase(tree, &bx, branchIdx, 0, lftBox);
	
	if(stat) return stat;
		
	if(bx.getMax(axis) > splitPos) 
		stat = innerBroadphase(tree, &bx, branchIdx, 1, rgtBox);
		
	return stat;
}

}
//:~