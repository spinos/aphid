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
#include <RayIntersectionContext.h>
#include <IntersectionContext.h>
#include <QElapsedTimer>

KdTree::KdTree() 
{
	m_root = new KdTreeNode;
}

KdTree::~KdTree() 
{
	delete m_root;
}

KdTreeNode* KdTree::getRoot() const
{ 
	return m_root; 
}

void KdTree::addMesh(BaseMesh* mesh)
{
	unsigned nf = mesh->getNumFaces();
	printf("add %i triangles\n", nf);
	m_stream.appendMesh(mesh);

	const BoundingBox box = mesh->calculateBBox();
	m_bbox.expandBy(box);
}

void KdTree::create()
{	
    printf("input primitive count %d\n", m_stream.getNumPrimitives());
	printf("tree bbox: %f %f %f - %f %f %f\n", m_bbox.getMin(0), m_bbox.getMin(1), m_bbox.getMin(2), m_bbox.getMax(0), m_bbox.getMax(1), m_bbox.getMax(2));
	
	QElapsedTimer timer;
	timer.start();
	
	BuildKdTreeContext *ctx = new BuildKdTreeContext(m_stream);
	ctx->setBBox(m_bbox);
	
	subdivide(m_root, *ctx, 0);
	ctx->verbose();
	delete ctx;
	
	m_stream.verbose();
	std::cout << "kd tree finished after " << timer.elapsed() << "ms\n";
}

void KdTree::subdivide(KdTreeNode * node, BuildKdTreeContext & ctx, int level)
{
	if(ctx.getNumPrimitives() < 30 || level == 22) {
		if(ctx.getNumPrimitives() > 0) {
			IndexArray &indir = m_stream.indirection();
			unsigned numDir = ctx.getNumPrimitives();
			node->setPrimStart(indir.index());
			node->setNumPrims(numDir);
			
			indir.expandBy(numDir);
			unsigned *src = ctx.indices();
			for(unsigned i = 0; i < numDir; i++) {
				unsigned *idx = indir.asIndex();
				*idx = src[i];
				indir.next();
			}
		}
		node->setLeaf(true);
		
		return;
	}
	
	//printf("subdiv node level %i\n", level);
	
	KdTreeBuilder builder(ctx);

	const SplitEvent *plane = builder.bestSplit();
	
	//plane->verbose();
	
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
	else
		branch->leaf.combined = 6;
		
	delete leftCtx;

	if(plane->rightCount() > 0)
		subdivide(branch + 1, *rightCtx, level + 1);
	else
		(branch+1)->leaf.combined = 6;
		
	delete rightCtx;
}

char KdTree::intersect(const Ray &ray, RayIntersectionContext * ctx)
{
	float hitt0, hitt1;
	if(!m_bbox.intersect(ray, &hitt0, &hitt1)) return 0;
	
	ctx->setBBox(m_bbox);

	KdTreeNode * root = getRoot();
	return recusiveIntersect(root, ray, ctx);
}

char KdTree::recusiveIntersect(KdTreeNode *node, const Ray &ray, RayIntersectionContext * ctx)
{
	//printf("recus intersect level %i\n", ctx.m_level);
	if(node->isLeaf()) {
		return leafIntersect(node, ray, ctx);
	}
	const int axis = node->getAxis();
	
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
		if(recusiveIntersect(nearNode, ray, ctx)) return 1;
	
		if(tplane < ray.m_tmin || tplane > ray.m_tmax)
			return 0;
		
		ctx->setBBox(farBox);
		ctx->m_level--;
		if(recusiveIntersect(farNode, ray, ctx)) return 1;
	}
	else {
		if(tplane > 0) {
			float hitt0, hitt1;
			bigBox.intersect(ray, &hitt0, &hitt1);
			if(tplane > hitt1) {
				ctx->setBBox(nearBox);
				ctx->m_level++;
				if(recusiveIntersect(nearNode, ray, ctx)) return 1;
			}
			else {
				ctx->setBBox(farBox);
				ctx->m_level++;
				if(recusiveIntersect(farNode, ray, ctx)) return 1;
			}
		}
		else {
				ctx->setBBox(nearBox);
				ctx->m_level++;
				if(recusiveIntersect(nearNode, ray, ctx)) return 1;
		}
	}
	return 0;
}

char KdTree::leafIntersect(KdTreeNode *node, const Ray &ray, RayIntersectionContext * ctx)
{
	unsigned start = node->getPrimStart();
	unsigned num = node->getNumPrims();
	//printf("prim start %i ", start);
		
	//printf("prims count in leaf %i start at %i\n", node->getNumPrims(), node->getPrimStart());
	IndexArray &indir = m_stream.indirection();
	PrimitiveArray &prims = m_stream.primitives();
	indir.setIndex(start);
	char anyHit = 0;
	for(unsigned i = 0; i < num; i++) {
		unsigned *iprim = indir.asIndex();

		Primitive * prim = prims.asPrimitive(*iprim);
		BaseMesh *mesh = (BaseMesh *)prim->getGeometry();
		unsigned iface = prim->getComponentIndex();
		
		if(mesh->intersect(iface, ray, ctx)) {
			//ctx->m_primitive = prim;
			anyHit = 1;
			//printf("hit %i\n", iface);
		}
		//else
		    //printf("miss %i\n", iface);
			
		indir.next();
	}
	if(anyHit) {ctx->m_success = 1; ctx->m_cell = (char *)node;}
	return anyHit;
}

Primitive * KdTree::getPrim(unsigned idx)
{
    IndexArray &indir = m_stream.indirection();
	PrimitiveArray &prims = m_stream.primitives();
	indir.setIndex(idx);
	unsigned *iprim = indir.asIndex();
	return  prims.asPrimitive(*iprim);
}

char KdTree::closestPoint(const Vector3F & origin, IntersectionContext * ctx)
{
	KdTreeNode * root = getRoot();
	ctx->setBBox(m_bbox);
	return recusiveClosestPoint(root, origin, ctx);
}

char KdTree::recusiveClosestPoint(KdTreeNode *node, const Vector3F &origin, IntersectionContext * ctx)
{
	int level = ctx->m_level;
	level++;
	if(node->isLeaf()) {
		return leafClosestPoint(node, origin, ctx);
	}
	const int axis = node->getAxis();
	const float splitPos = node->getSplitPos();
	BoundingBox leftBox, rightBox;
	BoundingBox bigBox = ctx->getBBox();
	bigBox.split(axis, splitPos, leftBox, rightBox);
	KdTreeNode *nearNode, *farNode;
	BoundingBox nearBox, farBox;
	nearNode = node->getLeft();
	farNode = node->getRight();
	nearBox = leftBox;
	farBox = rightBox;
	
	char hit = 0;
	if(nearBox.isPointAround(origin, ctx->m_minHitDistance)) {
		ctx->setBBox(nearBox);
		ctx->m_level = level;
		hit = recusiveClosestPoint(nearNode, origin, ctx);
	}

	if(farBox.isPointAround(origin, ctx->m_minHitDistance)) {
		ctx->setBBox(farBox);
		ctx->m_level = level;
		hit = recusiveClosestPoint(farNode, origin, ctx);
		
	}

	return hit;
}

char KdTree::leafClosestPoint(KdTreeNode *node, const Vector3F &origin, IntersectionContext * ctx)
{
	unsigned start = node->getPrimStart();
	unsigned num = node->getNumPrims();
	
	IndexArray &indir = m_stream.indirection();
	PrimitiveArray &prims = m_stream.primitives();
	indir.setIndex(start);
	char anyHit = 0;
	for(unsigned i = 0; i < num; i++) {
		unsigned *iprim = indir.asIndex();

		Primitive * prim = prims.asPrimitive(*iprim);
		BaseMesh *mesh = (BaseMesh *)prim->getGeometry();
		unsigned iface = prim->getComponentIndex();
		
		if(mesh->closestPoint(iface, origin, ctx)) {
			anyHit = 1;
		}
			
		indir.next();
	}
	if(anyHit) {ctx->m_success = 1; ctx->m_cell = (char *)node;}
	return anyHit;
}
