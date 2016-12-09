/*
 *  KdTree.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "KdTree.h"
#include <Ray.h>
#include <IntersectionContext.h>
#include <geom/ClosestToPointTest.h>
#include "SelectionContext.h"
#include <boost/timer.hpp>
#include <iostream>
#include <sstream>

namespace aphid {

KdTree::KdTree() 
{
	m_root = 0;
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

void KdTree::create(BuildProfile * prof)
{	
	KdTreeBuilder::MaxLeafPrimThreashold = prof->_maxLeafPrims;
	KdTreeBuilder::MaxBuildLevel = prof->_maxLevel;
	
	m_root = (KdTreeNode *)malloc(sizeof(KdTreeNode) * 2);
	
	const BoundingBox & b = getBBox();
    /// std::cout<<"\n Kd tree level 0 box "<<b.str();
	boost::timer bTimer;
	bTimer.restart();
	
	BuildKdTreeContext *ctx = new BuildKdTreeContext(m_stream, b);
	
	std::cout<<"\n Kd tree building...";
	std::stringstream sst;
	sst << "\n Kd tree pre-built in " << bTimer.elapsed();
    
	bTimer.restart();
	
	resetPropery();
	setTotalVolume(b.volume() );
	
	BuildKdTreeContext::GlobalContext = ctx;
	
	subdivide(m_root, *ctx, 0);
	//ctx->verbose();
	delete ctx;
	
	m_stream.removeInput();
    sst << "\n Kd tree built in " << bTimer.elapsed() << " secs"
	<<"\n total num nodes "<<m_stream.numNodes()
	<<"\n num prim indirections "<<m_stream.numIndirections()
	<<logProperty();
    m_buildLogStr = sst.str();
    std::cout<<m_buildLogStr;
}

void KdTree::clear()
{
	if(m_root) delete m_root;
	m_root = 0;
	m_stream.cleanup();
}

void KdTree::subdivide(KdTreeNode * node, BuildKdTreeContext & ctx, int level)
{
	addMaxLevel(level);
	if(ctx.numPrims() < KdTreeBuilder::MaxLeafPrimThreashold 
		|| level == KdTreeBuilder::MaxBuildLevel) {
		createLeaf(node, ctx);
		return;
	}
	
	KdTreeBuilder builder;
	
	builder.setContext(ctx);

	const SplitEvent *plane = builder.bestSplit();
	
	if(plane->getCost() > ctx.visitCost()) {
		createLeaf(node, ctx);
		return;
	}
	
	node->setAxis(plane->getAxis());
	node->setSplitPos(plane->getPos());
	KdTreeNode* branch = m_stream.createTreeBranch();
	
	node->setLeft(branch);
	node->setInternal();
	addNInternal();

	BuildKdTreeContext *leftCtx = new BuildKdTreeContext();
	BuildKdTreeContext *rightCtx = new BuildKdTreeContext();
	
	builder.partition(*leftCtx, *rightCtx);
	
	if(plane->leftCount() > 0)
		subdivide(branch, *leftCtx, level + 1);
	else {
		branch->leaf.combined = 6;
		branch->setNumPrims(0);
		addEmptyVolume(leftCtx->getBBox().volume() );
	}
		
	delete leftCtx;

	if(plane->rightCount() > 0)
		subdivide(branch + 1, *rightCtx, level + 1);
	else {
		(branch+1)->leaf.combined = 6;
		(branch+1)->setNumPrims(0);
		addEmptyVolume(rightCtx->getBBox().volume() );
	}
		
	delete rightCtx;
}

void KdTree::createLeaf(KdTreeNode * node, BuildKdTreeContext & ctx)
{
	ctx.decompressPrimitives(true);
	node->setLeaf();
	if(ctx.numPrims() < 1) {
		addEmptyVolume(ctx.getBBox().volume() );
		return;
	}
	const int numDir = ctx.numPrims();
	
	const sdb::VectorArray<Primitive> &prim = m_stream.primitives();
	sdb::VectorArray<Primitive> &indir = m_stream.indirection();
	
	node->setPrimStart(indir.size());
	
	const sdb::VectorArray<unsigned> & src = ctx.indices();
	for(int i = 0; i < numDir; i++)
		indir.insert(*prim[*src[i]]);
	
	addNLeaf();
	updateNPrim(numDir);
	
	node->setNumPrims(numDir);
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
	const int num = node->getNumPrims();
	if(num < 1) return 0;
	const int start = node->getPrimStart();
    // std::cout<<"\n leaf intersect"<<start<<" "<<num;
	
	sdb::VectorArray<Primitive> &indir = m_stream.indirection();
	int igeom, icomponent;
	char anyHit = 0;
	for(unsigned i = 0; i < num; i++) {
		//unsigned *iprim = indir.asIndex();
		//unsigned * iprim = indir[start + i];
		//Primitive * prim = prims.get(*iprim);
		Primitive * prim = indir[start + i];
		prim->getGeometryComponent(igeom, icomponent);
		Geometry * geo = m_stream.geometry(igeom);
		
		if(geo->intersectRay(icomponent, &ctx->m_ray, ctx->m_hitP, ctx->m_hitN, ctx->m_minHitDistance) ) {
			anyHit = 1;
			ctx->m_geometry = geo;
			ctx->m_componentIdx = icomponent;
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
    sdb::VectorArray<Primitive> &indir = m_stream.indirection();
	return indir[idx];
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
	const int num = node->getNumPrims();
	if(num < 1) return 0;
	const int start = node->getPrimStart();
	sdb::VectorArray<Primitive> &indir = m_stream.indirection();
	int igeom, icomponent;
	for(int i = 0; i < num; i++) {
		Primitive * prim = indir[start + i];
		prim->getGeometryComponent(igeom, icomponent);
		Geometry * geo = m_stream.geometry(igeom);
		
		if(geo->intersectSphere(icomponent, ctx->sphere() ) )
			ctx->select(geo, icomponent);
	}
	return 1;
}

const TypedEntity::Type KdTree::type() const
{ return TypedEntity::TKdTree; }

sdb::VectorArray<Primitive> & KdTree::indirection()
{ return m_stream.indirection(); }

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
	const int num = node->getNumPrims();
	if(num < 1) return;
	const int start = node->getPrimStart();
	sdb::VectorArray<Primitive> &indir = indirection();
	///sdb::VectorArray<Primitive> &prims = primitives();
	int igeom, icomponent;
	for(int i = 0; i < num; i++) {
		//unsigned *iprim = indir[start + i];
		//Primitive * prim = prims.get(*iprim);
		Primitive * prim = indir[start + i];
		prim->getGeometryComponent(igeom, icomponent);
		Geometry * geo = m_stream.geometry(igeom);
		
		geo->closestToPoint(icomponent, result);
	}
}

bool KdTree::intersectBox(const BoundingBox & box)
{
	KdTreeNode * root = getRoot();
	if(!root) return false;
	
	BoundingBox b = getBBox();
	
	m_testBox = box;
	
	return recursiveIntersectBox(root, b);
}

bool KdTree::recursiveIntersectBox(KdTreeNode *node, const BoundingBox & box)
{
	if(!box.intersect(m_testBox)) return false;
	
	if(node->isLeaf())
		return leafIntersectBox(node, box);
		
	const int axis = node->getAxis();
	const float splitPos = node->getSplitPos();
	
	BoundingBox leftBox, rightBox;
	box.split(axis, splitPos, leftBox, rightBox);
	
	if(recursiveIntersectBox(node->getLeft(), leftBox)) return true;
	
	return recursiveIntersectBox(node->getRight(), rightBox);
}

bool KdTree::leafIntersectBox(KdTreeNode *node, const BoundingBox & box)
{
	const int num = node->getNumPrims();
	if(num < 1) return false;
	
	const int start = node->getPrimStart();
	sdb::VectorArray<Primitive> &indir = indirection();
	//sdb::VectorArray<Primitive> &prims = primitives();
	int igeom, icomponent;
	for(int i = 0; i < num; i++) {
		//unsigned *iprim = indir[start + i];
		//Primitive * prim = prims.get(*iprim);
		Primitive * prim = indir[start + i];
		prim->getGeometryComponent(igeom, icomponent);
		Geometry * geo = m_stream.geometry(igeom);
		
		if(geo->intersectBox(icomponent, m_testBox)) {
			m_intersectElement = icomponent;
			return true;
		}
		//indir.next();
	}
	return false;
}

std::string KdTree::buildLog() const
{ return m_buildLogStr; }

}
//:~