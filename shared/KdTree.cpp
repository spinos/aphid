/*
 *  KdTree.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <APhid.h>
#include "KdTree.h"
#include <Ray.h>
#include <IntersectionContext.h>
#include "SelectionContext.h"
#include <boost/timer.hpp>

int KdTree::MaxBuildLevel = 32;
unsigned KdTree::NumPrimitivesInLeafThreashold = 32;

KdTree::KdTree() 
{
	m_root = 0;
	m_maxLeafLevel = 0;
	m_numNoEmptyLeaf = 0;
}

KdTree::~KdTree() 
{
	clear();
}

bool KdTree::isEmpty() const
{
	return m_root == 0;
}

KdTreeNode* KdTree::getRoot() const
{ 
	return m_root; 
}

void KdTree::addGeometry(Geometry * geo)
{
	m_stream.appendGeometry(geo);
	updateBBox(geo->calculateBBox());
}

void KdTree::create()
{	
	m_root = new KdTreeNode;
	
	const BoundingBox & b = getBBox();
    std::cout<<"\n Kd tree level 0 box "<<b.str();
	boost::timer bTimer;
	bTimer.restart();
	
	BuildKdTreeContext *ctx = new BuildKdTreeContext(m_stream, b);
	
	std::cout << "\n Kd tree prepare in " << bTimer.elapsed();
	bTimer.restart();
	
	m_maxLeafLevel = 0;
	m_numNoEmptyLeaf = 0;
	
	KdTreeBuilder::GlobalContext = ctx;
	m_minNumLeafPrims = 1e28;
	m_maxNumLeafPrims = 0;
	m_totalNumLeafPrims = 0;
	
	subdivide(m_root, *ctx, 0);
	ctx->verbose();
	delete ctx;
	
	// m_stream.verbose();
	std::cout << "\n Kd tree built in " << bTimer.elapsed() << " secs"
	<<"\n total num nodes "<<m_stream.numNodes()
	<<"\n max leaf level: "<<m_maxLeafLevel
	<<"\n num no-empty leaves "<<m_numNoEmptyLeaf
	<<"\n min/max leaf prims "<<m_minNumLeafPrims<<"/"<<m_maxNumLeafPrims
	<<"\n average "<<(float)m_totalNumLeafPrims/(float)m_numNoEmptyLeaf;
}

void KdTree::rebuild()
{
	clear();
	m_stream.initialize();
	create();
}

void KdTree::clear()
{
	if(m_root) delete m_root;
	m_root = 0;
	m_stream.cleanup();
}

void KdTree::subdivide(KdTreeNode * node, BuildKdTreeContext & ctx, int level)
{
	if(ctx.getNumPrimitives() < NumPrimitivesInLeafThreashold 
		|| level == KdTree::MaxBuildLevel) {
		/// if(ctx.isCompressed()) std::cout<<"\n still compressed "<<level;
		if(level > m_maxLeafLevel) m_maxLeafLevel = level;
		createLeaf(node, ctx);
		return;
	}
	
	KdTreeBuilder builder;
	
	builder.setContext(ctx);

	const SplitEvent *plane = builder.bestSplit();
	
	if(plane->getCost() > ctx.visitCost()) {
		/// if(ctx.isCompressed()) std::cout<<"\n still compressed "<<level;
		if(level > m_maxLeafLevel) m_maxLeafLevel = level;
		createLeaf(node, ctx);
		return;
	}
	
	node->setAxis(plane->getAxis());
	node->setSplitPos(plane->getPos());
	KdTreeNode* branch = m_stream.createTreeBranch();
	
	node->setLeft(branch);
	node->setLeaf(false);

	BuildKdTreeContext *leftCtx = new BuildKdTreeContext();
	BuildKdTreeContext *rightCtx = new BuildKdTreeContext();
	
	builder.partition(*leftCtx, *rightCtx);
	
	if(plane->leftCount() > 0)
		subdivide(branch, *leftCtx, level + 1);
	else {
		branch->leaf.combined = 6;
		branch->setNumPrims(0);
	}
		
	delete leftCtx;

	if(plane->rightCount() > 0)
		subdivide(branch + 1, *rightCtx, level + 1);
	else {
		(branch+1)->leaf.combined = 6;
		(branch+1)->setNumPrims(0);
	}
		
	delete rightCtx;
}

void KdTree::createLeaf(KdTreeNode * node, BuildKdTreeContext & ctx)
{
	ctx.decompress(true);
	if(!ctx.grid() ) {
	if(ctx.getNumPrimitives() > 0) {
		const unsigned numDir = ctx.getNumPrimitives();
		
		if(m_minNumLeafPrims > numDir )
			m_minNumLeafPrims = numDir;
		if(m_maxNumLeafPrims < numDir )
			m_maxNumLeafPrims = numDir;
		m_totalNumLeafPrims += numDir;
		
		std::vector<unsigned> &indir = m_stream.indirection();
		node->setPrimStart(indir.size());
		node->setNumPrims(numDir);
		
		//indir.expandBy(numDir);
		unsigned *src = ctx.indices();
		for(unsigned i = 0; i < numDir; i++) {
			//unsigned *idx = indir.asIndex();
			//*idx = src[i];
			//indir.next();
			indir.push_back(src[i]);
		}
		m_numNoEmptyLeaf++;
	}
	}
	
	node->setLeaf(true);
}

char KdTree::intersect(IntersectionContext * ctx)
{
	if(!getRoot()) return 0;
	float hitt0, hitt1;
	BoundingBox b = getBBox();
	if(!b.intersect(ctx->m_ray, &hitt0, &hitt1)) return 0;
	
	ctx->setBBox(b);

	KdTreeNode * root = getRoot();
	return recusiveIntersect(root, ctx);
}

char KdTree::recusiveIntersect(KdTreeNode *node, IntersectionContext * ctx)
{
	if(node->isLeaf())
		return leafIntersect(node, ctx);
	
	const int axis = node->getAxis();
	const Ray & ray = ctx->m_ray;
	const float splitPos = node->getSplitPos();
	const float invRayDir = 1.f / ray.m_dir.comp(axis);
	const Vector3F o = ray.m_origin;
	const float origin = ray.m_origin.comp(axis);
	char belowPlane = (origin < splitPos || (origin == splitPos && ray.m_dir.comp(axis) <= 0.f));
	
	BoundingBox leftBox, rightBox;
	BoundingBox bigBox = ctx->getBBox();
	bigBox.split(axis, splitPos, leftBox, rightBox);
	
	KdTreeNode *nearNode, *farNode;
	BoundingBox nearBox, farBox;
	if(belowPlane) {
		nearNode = node->getLeft();
		farNode = node->getRight();
		nearBox = leftBox;
		farBox = rightBox;
	}
	else {
		farNode = node->getLeft();
		nearNode = node->getRight();
		farBox = leftBox;
		nearBox = rightBox;
	}
	float tplane = (splitPos - origin) * invRayDir;
	Vector3F pplane = ray.m_origin + ray.m_dir * tplane;

	if(bigBox.isPointInside(pplane)) {
		ctx->setBBox(nearBox);
		ctx->m_level++;
		if(recusiveIntersect(nearNode, ctx)) return 1;
	
		if(tplane < ray.m_tmin || tplane > ray.m_tmax)
			return 0;
		
		ctx->setBBox(farBox);
		ctx->m_level--;
		if(recusiveIntersect(farNode, ctx)) return 1;
	}
	else {
		if(tplane > 0) {
			float hitt0, hitt1;
			bigBox.intersect(ray, &hitt0, &hitt1);
			if(tplane > hitt1) {
				ctx->setBBox(nearBox);
				ctx->m_level++;
				if(recusiveIntersect(nearNode, ctx)) return 1;
			}
			else {
				ctx->setBBox(farBox);
				ctx->m_level++;
				if(recusiveIntersect(farNode, ctx)) return 1;
			}
		}
		else {
				ctx->setBBox(nearBox);
				ctx->m_level++;
				if(recusiveIntersect(nearNode, ctx)) return 1;
		}
	}
	return 0;
}

char KdTree::leafIntersect(KdTreeNode *node, IntersectionContext * ctx)
{
	const unsigned num = node->getNumPrims();
	if(num < 1) return 0;
	unsigned start = node->getPrimStart();
	//printf("prim start %i ", start);
	//printf("prims count in leaf %i start at %i\n", node->getNumPrims(), node->getPrimStart());
	std::vector<unsigned> &indir = m_stream.indirection();
	sdb::VectorArray<Primitive> &prims = m_stream.primitives();
	//indir.setIndex(start);
	char anyHit = 0;
	float hitD;
	for(unsigned i = 0; i < num; i++) {
		//unsigned *iprim = indir.asIndex();
		unsigned iprim = indir[start + i];
		Primitive * prim = prims.get(iprim);
		Geometry * geo = prim->getGeometry();
		unsigned icomp = prim->getComponentIndex();
		
		if(geo->intersectRay(icomp, &ctx->m_ray, ctx->m_hitP, ctx->m_hitN, ctx->m_minHitDistance) ) {
			anyHit = 1;
			ctx->m_geometry = geo;
			ctx->m_componentIdx = icomp;
		}
			
		//indir.next();
	}
	if(anyHit) {
		ctx->m_success = 1; 
		ctx->m_cell = (char *)node;
	}
	return anyHit;
}

Primitive * KdTree::getPrim(unsigned idx)
{
    std::vector<unsigned> &indir = m_stream.indirection();
	sdb::VectorArray<Primitive> &prims = m_stream.primitives();	
	return  prims.get(indir[idx]);
}

void KdTree::select(SelectionContext * ctx)
{
	KdTreeNode * root = getRoot();
	if(!root) return;
	
	const BoundingBox b = getBBox();
	if(!ctx->closeTo(b)) return;
	
	ctx->setBBox(b);

	recursiveSelect(root, ctx);
}

char KdTree::recursiveSelect(KdTreeNode *node, SelectionContext * ctx)
{
	if(node->isLeaf())
		return leafSelect(node, ctx);
		
	const int axis = node->getAxis();
	const float splitPos = node->getSplitPos();
	
	BoundingBox leftBox, rightBox;
	const BoundingBox bigBox = ctx->getBBox();
	bigBox.split(axis, splitPos, leftBox, rightBox);
	
	if(ctx->closeTo(leftBox)) {
		ctx->setBBox(leftBox);
		recursiveSelect(node->getLeft(), ctx);
	}
	
	if(ctx->closeTo(rightBox)) {
		ctx->setBBox(rightBox);
		recursiveSelect(node->getRight(), ctx);
	}
	
	return 1;
}

char KdTree::leafSelect(KdTreeNode *node, SelectionContext * ctx)
{
	const unsigned num = node->getNumPrims();
	if(num < 1) return 0;
	unsigned start = node->getPrimStart();
	std::vector<unsigned> &indir = m_stream.indirection();
	sdb::VectorArray<Primitive> &prims = m_stream.primitives();

	for(unsigned i = 0; i < num; i++) {
		unsigned iprim = indir[start + i];
		
		Primitive * prim = prims.get(iprim);
		Geometry * geo = prim->getGeometry();
		unsigned icomponent = prim->getComponentIndex();
		
		if(geo->intersectSphere(icomponent, ctx->sphere() ) )
			ctx->select(geo, icomponent);
	}
	return 1;
}

const unsigned KdTree::numNoEmptyLeaves() const
{ return m_numNoEmptyLeaf; }

const TypedEntity::Type KdTree::type() const
{ return TypedEntity::TKdTree; }

std::vector<unsigned> & KdTree::indirection()
{ return m_stream.indirection(); }

sdb::VectorArray<Primitive> & KdTree::primitives()
{ return m_stream.primitives(); }

void KdTree::closestToPoint(ClosestToPointTestResult * result)
{ 
	if(result->closeEnough() ) return;
	recusiveClosestToPoint(getRoot(), getBBox(), result); 
}

void KdTree::recusiveClosestToPoint(KdTreeNode *node, const BoundingBox &box, ClosestToPointTestResult * result)
{
	if(!result->closeTo(box)) return;
	if(node->isLeaf())
		return leafClosestToPoint(node, box, result);
		
	const int axis = node->getAxis();
	const float splitPos = node->getSplitPos();
	BoundingBox leftBox, rightBox;
	box.split(axis, splitPos, leftBox, rightBox);
	
	const float cp = result->_toPoint.comp(axis) - splitPos;
	if(cp < 0.f) {
		recusiveClosestToPoint(node->getLeft(), leftBox, result);
		if(result->closeEnough() ) return;
		if( -cp < result->_distance) 
			recusiveClosestToPoint(node->getRight(), rightBox, result);
	}
	else {
		recusiveClosestToPoint(node->getRight(), rightBox, result);
		if(result->closeEnough() ) return;
		if(cp < result->_distance)
			recusiveClosestToPoint(node->getLeft(), leftBox, result);
	}
	
}

void KdTree::leafClosestToPoint(KdTreeNode *node, const BoundingBox &box, ClosestToPointTestResult * result)
{
	const unsigned num = node->getNumPrims();
	if(num < 1) return;
	
	const unsigned start = node->getPrimStart();
	std::vector<unsigned> &indir = indirection();
	sdb::VectorArray<Primitive> &prims = primitives();
	
	for(unsigned i = 0; i < num; i++) {
		unsigned iprim = indir[start + i];
		Primitive * prim = prims.get(iprim);
		Geometry * geo = prim->getGeometry();
		unsigned icomponent = prim->getComponentIndex();
		
		geo->closestToPoint(icomponent, result);
	}
}
//:~