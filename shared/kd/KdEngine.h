/*
 *  KdEngine.h
 *  testntree
 *
 *  Created by jian zhang on 11/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <kd/KdNTree.h>
#include <kd/KdBuilder.h>
#include <IntersectionContext.h>
#include <geom/SelectionContext.h>
#include <geom/ClosestToPointTest.h>
#include <geom/ConvexShape.h>

namespace aphid {

class KdEngine {

    int m_numRopeTraversed;
    
public:
	template<typename T, int Ies>
	void buildSource(sdb::VectorArray<T> * dst,
					BoundingBox & box,
					const float * points,
					const int & numIndices,
					const int * elementIndices);
					
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
	void broadphaseSelect(KdNTree<T, Tn > * tree, 
				SphereSelectionContext * ctx);
				
	template<typename T, typename Tn>
	void closestToPoint(KdNTree<T, Tn > * tree, 
				ClosestToPointTestResult * ctx);
				
	template<typename T, typename Tn>
	void intersectBox(KdNTree<T, Tn > * tree, 
				BoxIntersectContext * ctx);

/// domain of element much smaller than bx
	template<typename T, typename Tn>
	bool broadphase(KdNTree<T, Tn > * tree,
				const BoundingBox & bx);
				
	template<typename T, typename Tn>
	bool beamIntersect(KdNTree<T, Tn > * tree, 
				IntersectionContext * ctx);
				
	template<typename T, typename Tn>
	bool narrowphase(KdNTree<T, Tn > * tree, 
				const cvx::Hexagon & hexa);			
	
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
	
/// 0 or 1 side of split, -1 no hit
	template <typename T>
	int firstVisit(IntersectionContext * ctx, 
					const KdTreeNode * n);
	
/// -1 branch
/// 1 leaf
/// 0 empty
	template <typename T, typename Tn>
	int visitBranchOrLeaf(IntersectionContext * ctx, 
					int & branchIdx,
					int & nodeIdx,
					const KdTreeNode * r);
					
	template <typename T, typename Tn>
	int rayPrimitive(KdNTree<T, Tn > * tree, 
							IntersectionContext * ctx, 
							const KdTreeNode * r);
	
	template <typename T, typename Tn>
	int beamPrimitive(KdNTree<T, Tn > * tree, 
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
								ClosestToPointTestResult * result,
								KdTreeNode *node, const BoundingBox &box);
								
	template<typename T, typename Tn>
	void innerClosestToPoint(KdNTree<T, Tn > * tree, 
				ClosestToPointTestResult * ctx,
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
				
	template<typename T, typename Tn>
	bool leafNarrowphase(KdNTree<T, Tn > * tree,
				const cvx::Hexagon & hexa,
				const BoundingBox & b,
				KdTreeNode * r);
				
	template <typename T, typename Tn>
	bool innerNarrowphase(KdNTree<T, Tn > * tree,
				const cvx::Hexagon & hexa,
				const BoundingBox & b,
				int branchIdx,
				int nodeIdx,
				const BoundingBox & innerBx);
				
	template<typename T, typename Tn>
	void broadphaseLeafSelect(KdNTree<T, Tn > * tree, 
					SphereSelectionContext * ctx,
					KdTreeNode * r);
	
	template<typename T, typename Tn>
	void broadphaseInnerSelect(KdNTree<T, Tn > * tree, 
					SphereSelectionContext * ctx,
					int branchIdx,
					int nodeIdx,
					const BoundingBox & b);
				
};

template<typename T, int Ies>
void KdEngine::buildSource(sdb::VectorArray<T> * dst,
					BoundingBox & box,
					const float * points,
					const int & numIndices,
					const int * elementIndices)
{
	T acomp;
	box.reset();
	for(int i=0;i<numIndices;i+=Ies) {
		const int * ind = &elementIndices[i];
		for(int j=0;j<Ies;++j) {
			const int & vj = ind[j];
			const Vector3F pj(points[vj*3], points[vj*3 + 1], points[vj*3 + 2]);
			box.expandBy(pj, 1e-4f);
			acomp.setP(pj, j);
			acomp.setInd(i/Ies, 1);
			
		}
		dst->insert(acomp);
	}
	box.round();
}

template<typename T, typename Ts>
void KdEngine::buildSource(sdb::VectorArray<T> * dst,
					BoundingBox & box,
					const std::vector<Ts *> & src)
{
	box.reset();
	T acomp;
	typename std::vector<Ts *>::const_iterator it = src.begin();
    for(int i=0;it!=src.end();++it,++i) {
/// ind 0 to geom
		acomp.setInd(i, 0);
		const Ts * w = *it;
		const int n = w->numComponents();
		for(int j=0; j<n; ++j) {
		
			w-> template dumpComponent<T>(acomp, j);
			
/// ind 1 to component
			acomp.setInd(j, 1);
	
			dst->insert(acomp);
			
			const BoundingBox cbx = acomp.calculateBBox();
			box.expandBy(cbx);
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
			<<"\n n input "<<source->size()
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
	if(tree->isEmpty()) {
		std::cout<<" KdEngine intersect null tree"<<std::endl;
		return 0;
	}
	
	const BoundingBox & b = tree->getBBox();
	if(!b.intersect(ctx->m_ray)) {
		std::cout<<" KdEngine intersect oob"<<b<<std::endl;
		return 0;
	}
	
	m_numRopeTraversed = 0;
    
	ctx->setBBox(b);
	
	const KdTreeNode * r = tree->root()->node(0);
	if(r->isLeaf() ) {
		ctx->m_leafIdx = r->getPrimStart();
		rayPrimitive(tree, ctx, r);
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
		stat = visitBranchOrLeaf<T, Tn>(ctx, branchIdx, nodeIdx, 
							kn);
							
		if(preBranchIdx != branchIdx) {
			currentBranch = tree->branches()[branchIdx];
			preBranchIdx = branchIdx;
		}
		
		kn = currentBranch->node(nodeIdx);
		
		if(stat > 0 ) {
			if(rayPrimitive(tree, ctx, kn ) ) hasNext = false;
			else stat = 0;
		}
		
		if(ctx->m_ray.length() < 1e-3) {
			return true;
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
    const BoundingBox & b = ctx->getBBox();
	if(!b.intersect(ctx->m_ray, &ctx->m_tmin, &ctx->m_tmax) )
		return -1;
	
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
}

template <typename T, typename Tn>
int KdEngine::visitBranchOrLeaf(IntersectionContext * ctx, 
									int & branchIdx,
									int & nodeIdx,
									const KdTreeNode * r)
{
//	std::cout<<"\n node "<<nodeIdx;
				
	if(r->isLeaf() ) {
//		std::cout<<"\n hit leaf "<<r->getPrimStart();
		ctx->m_leafIdx = r->getPrimStart();
		if(r->getNumPrims() < 1)
			return 0;
		
		return 1;
	}
	
	int fv; 
	
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		// std::cout<<"\n inner offset "<<offset;
		fv = firstVisit<T>(ctx, r);
		if(fv < 0)
			return 0;
			
		nodeIdx += offset + fv;
	}
	else {
		branchIdx += offset & Tn::TreeletOffsetMaskTau;
//		std::cout<<"\n branch "<<branchIdx;
		fv = firstVisit<T>(ctx, r);
		if(fv < 0)
			return 0;
			
		nodeIdx = fv;
	}
	
	return -1;
}

template <typename T, typename Tn>
int KdEngine::rayPrimitive(KdNTree<T, Tn > * tree, 
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
		if(c->rayIntersect(ctx->m_ray, &ctx->m_tmin, &ctx->m_tmax) ) {
			ctx->m_hitP = ctx->m_ray.travel(ctx->m_tmin);
			ctx->m_hitN = c->calculateNormal();
/// shorten ray
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
int KdEngine::beamPrimitive(KdNTree<T, Tn > * tree, 
							IntersectionContext * ctx, 
							const KdTreeNode * r)
{
	int start, len;
	tree->leafPrimStartLength(start, len, r->getPrimStart() );

	int nhit = 0;
	int i = 0;
	for(;i<len;++i) {
		const T * c = tree->getSource(start + i);
		if(c->beamIntersect(ctx->m_beam, ctx->splatRadius(), &ctx->m_tmin, &ctx->m_tmax) ) {
			ctx->m_hitP = ctx->m_ray.travel(ctx->m_tmin);
			ctx->m_hitN = c->calculateNormal();
			ctx->m_ray.m_tmax = ctx->m_tmin;
/// shorten beam
			ctx->m_beam.setTmax(ctx->m_tmin);
			ctx->m_success = 1;
			ctx->m_componentIdx = tree->primIndirectionAt(start + i);
			nhit++;
		}
	}
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
    if(m_numRopeTraversed > 19) {
		std::cout<<"\n KdEngine::climbRope out of rope traverse "
			<<ctx->m_ray<<" "<<ctx->getBBox()<<std::endl;
        return false;
	}
    
	const BoundingBox & b = ctx->getBBox();
	float t0 = 0.f, t1 = 0.f;
	b.intersect(ctx->m_ray, &t0, &t1);
		
	Vector3F hit1 = ctx->m_ray.travel(t1 + ctx->m_tdelta);
/// end inside 
	if(b.isPointInside(hit1) ) {
		//std::cout<<"\n KdEngine::climbRope "<<hit1<<" out of "<<b<<std::endl;
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
	
	const BoundingBox * rp = tree->ropes()[ iRope ];
	BoxNeighbors::DecodeTreeletNodeHash(rp->m_padding1, KdNode4::BranchingFactor, 
					branchIdx, nodeIdx);
         
//    std::cout.flush();
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
				ClosestToPointTestResult * ctx)
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
		if(ctx->closeEnough() ) {
			return;
		}
		if( -cp < ctx->_distance) {
			innerClosestToPoint(tree, ctx, branchIdx, 1, rightBox);
		}
	}
	else {
		innerClosestToPoint(tree, ctx, branchIdx, 1, rightBox);
		if(ctx->closeEnough() ) {
			return;
		}
		if(cp < ctx->_distance) {
			innerClosestToPoint(tree, ctx, branchIdx, 0, leftBox);
		}
	}
	
}
	
template<typename T, typename Tn>
void KdEngine::innerClosestToPoint(KdNTree<T, Tn > * tree, 
				ClosestToPointTestResult * ctx,
				int branchIdx,
				int nodeIdx,
				const BoundingBox & b)
{
	if(!ctx->closeTo(b) ) {
		return;
	}
		
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
			
			if(ctx->closeEnough() ) {
				return;
			}
			
			if( -cp < ctx->_distance) {
				innerClosestToPoint(tree, ctx, 
							branchIdx, 
							nodeIdx + offset + 1, 
							rgtBox);
			}
		}
		else {
			innerClosestToPoint(tree, ctx, 
							branchIdx,
							nodeIdx + offset + 1,
							rgtBox);
							
			if(ctx->closeEnough() ) {
				return;
			}
			
			if(cp < ctx->_distance) {
				innerClosestToPoint(tree, ctx, 
							branchIdx,
							nodeIdx + offset,
							lftBox);
			}
		}
	}
	else {
		if(cp < 0.f ) {
			innerClosestToPoint(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
							
			if(ctx->closeEnough() ) {
				return;
			}
			
			if( -cp < ctx->_distance) {
				innerClosestToPoint(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
			}
		}
		else {
			innerClosestToPoint(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
							
			if(ctx->closeEnough() ) {
				return;
			}
			
			if(cp < ctx->_distance) {
				innerClosestToPoint(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
			}
		}
	}
}

template<typename T, typename Tn>
void KdEngine::leafClosestToPoint(KdNTree<T, Tn > * tree, 
								ClosestToPointTestResult * result,
								KdTreeNode *node, const BoundingBox &box)
{
	if(node->getNumPrims() < 1) {
		return;
	}
	
	int start, len;
	tree->leafPrimStartLength(start, len, node->getPrimStart() );
	int i = 0;
	for(;i<len;++i) {
		const T * c = tree->getSource(start + i );
		c-> template closestToPoint<ClosestToPointTestResult>(result);
	}
	
}


template<typename T, typename Tn>
void KdEngine::intersectBox(KdNTree<T, Tn > * tree, 
				BoxIntersectContext * ctx)
{
	if(tree->isEmpty() ) {
		return;
	}
	
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

template<typename T, typename Tn>
bool KdEngine::beamIntersect(KdNTree<T, Tn > * tree, 
				IntersectionContext * ctx)
{ 
	if(tree->isEmpty()) return 0;
	
	const BoundingBox & b = tree->getBBox();
	if(!b.intersect(ctx->m_ray)) return 0;
	
	m_numRopeTraversed = 0;
    
	ctx->setBBox(b);
	
	const KdTreeNode * r = tree->root()->node(0);
	if(r->isLeaf() ) {
		ctx->m_leafIdx = r->getPrimStart();
		rayPrimitive(tree, ctx, r);
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
		stat = visitBranchOrLeaf<T, Tn>(ctx, branchIdx, nodeIdx, 
							kn);
							
		if(preBranchIdx != branchIdx) {
			currentBranch = tree->branches()[branchIdx];
			preBranchIdx = branchIdx;
		}
		
		kn = currentBranch->node(nodeIdx);
		
		if(stat > 0 ) {
			if(beamPrimitive(tree, ctx, kn ) ) hasNext = false;
			else stat = 0;
		}
		
		if(ctx->m_ray.length() < 1e-3)
			return true;
			
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

template<typename T, typename Tn>
bool KdEngine::leafNarrowphase(KdNTree<T, Tn > * tree,
			const cvx::Hexagon & hexa,
			const BoundingBox & b,
			KdTreeNode * r)
{
	if(r->getNumPrims() < 1) return false;
	int start, len;
	tree->leafPrimStartLength(start, len, r->getPrimStart() );
	int i = 0;
	for(;i<len;++i) {
		const T * c = tree->getSource(start + i );
        if(b.intersect(c->calculateBBox() ) ) {
			if(c-> template exactIntersect <cvx::Hexagon > (hexa) )
				return true;
		}
	}
	return false;
}
			
template<typename T, typename Tn>
bool KdEngine::innerNarrowphase(KdNTree<T, Tn > * tree,
			const cvx::Hexagon & hexa,
			const BoundingBox & b,
			int branchIdx,
			int nodeIdx,
			const BoundingBox & innerBx)
{
	Tn * currentBranch = tree->branches()[branchIdx];
	KdTreeNode * r = currentBranch->node(nodeIdx);
	if(r->isLeaf() ) {
		return leafNarrowphase(tree, hexa, b, r);
	}
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	innerBx.split(axis, splitPos, lftBox, rgtBox);
	
	bool stat = false;
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		if(b.getMin(axis) < splitPos ) {
			stat = innerNarrowphase(tree, hexa, b, 
							branchIdx,
							nodeIdx + offset,
							lftBox);
			if(stat ) return stat;
		}
		
		if(b.getMax(axis) > splitPos ) {
			stat = innerNarrowphase(tree, hexa, b, 
							branchIdx,
							nodeIdx + offset + 1,
							rgtBox);
			if(stat ) return stat;
		}
	}
	else {
		if(b.getMin(axis) < splitPos ) {
			stat = innerNarrowphase(tree, hexa, b, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
			if(stat ) return stat;
		}
		
		if(b.getMax(axis) > splitPos ) {
			stat = innerNarrowphase(tree, hexa, b, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
			if(stat ) return stat;
		}
	}
	return stat;
}


template<typename T, typename Tn>
bool KdEngine::narrowphase(KdNTree<T, Tn > * tree, 
				const cvx::Hexagon & hexa)
{
	const BoundingBox bx = hexa.calculateBBox();
	KdTreeNode * r = tree->root()->node(0);
	if(r->isLeaf() )
		return leafNarrowphase(tree, hexa, bx, r);
	
	const BoundingBox b = tree->getBBox();
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	
	bool stat = false;
	int branchIdx = tree->root()->internalOffset(0);
	if(bx.getMin(axis) < splitPos) 
		stat = innerNarrowphase(tree, hexa, bx, branchIdx, 0, lftBox);
	
	if(stat) return stat;
		
	if(bx.getMax(axis) > splitPos) 
		stat = innerNarrowphase(tree, hexa, bx, branchIdx, 1, rgtBox);
		
	return stat;
}

template<typename T, typename Tn>
void KdEngine::broadphaseSelect(KdNTree<T, Tn > * tree, 
						SphereSelectionContext * ctx)
{
	if(tree->isEmpty() ) {
		return;
	}
	
	const BoundingBox b = tree->getBBox();
	if(!b.intersect(*ctx)) {
		return;
	}

	KdTreeNode * r = tree->root()->node(0);
	if(r->isLeaf() ) {
		broadphaseLeafSelect(tree, ctx, r);
		return;
	}
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
		
	int branchIdx = tree->root()->internalOffset(0);
	broadphaseInnerSelect(tree, ctx, branchIdx, 0, lftBox);
	broadphaseInnerSelect(tree, ctx, branchIdx, 1, rgtBox);
	
}

template<typename T, typename Tn>
void KdEngine::broadphaseLeafSelect(KdNTree<T, Tn > * tree, 
				SphereSelectionContext * ctx,
				KdTreeNode * r)
{
	if(r->getNumPrims() < 1) {
		return;
	}
	
	int start, len;
	tree->leafPrimStartLength(start, len, r->getPrimStart() );
	
	for(int i=0;i<len;++i) {
		const T * c = tree->getSource(start + i );
		if(c->calculateBBox().intersect(*ctx) ) {
			ctx->addPrim(tree->primIndirectionAt(start + i) );
		}
	}
}

template<typename T, typename Tn>
void KdEngine::broadphaseInnerSelect(KdNTree<T, Tn > * tree, 
				SphereSelectionContext * ctx,
				int branchIdx,
				int nodeIdx,
				const BoundingBox & b)
{
	Tn * currentBranch = tree->branches()[branchIdx];
	KdTreeNode * r = currentBranch->node(nodeIdx);
	if(r->isLeaf() ) {
		broadphaseLeafSelect(tree, ctx, r);
		return;
	}
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		if(ctx->getMin(axis) < splitPos ) {
			broadphaseInnerSelect(tree, ctx, 
							branchIdx,
							nodeIdx + offset,
							lftBox);
		}
		
		if(ctx->getMax(axis) > splitPos ) {
			broadphaseInnerSelect(tree, ctx, 
							branchIdx,
							nodeIdx + offset + 1,
							rgtBox);
		}
	}
	else {
		if(ctx->getMin(axis) < splitPos ) {
			broadphaseInnerSelect(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
		}
		
		if(ctx->getMax(axis) > splitPos ) {
			broadphaseInnerSelect(tree, ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
		}
	}
}

}
//:~