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
#include <QElapsedTimer>

KdTree::KdTree() 
{
	m_root = new KdTreeNode;
	
	printf("axis mask        %s\n", byte_to_binary(KdTreeNode::EInnerAxisMask));
	printf("type        mask %s\n", byte_to_binary(KdTreeNode::ETypeMask));
	printf("indirection mask %s\n", byte_to_binary(KdTreeNode::EIndirectionMask));
	printf("leaf offset mask %s\n", byte_to_binary(KdTreeNode::ELeafOffsetMask));
	/*
	printf("32 bit align mask %s\n", byte_to_binary(0xffffffff - 31));
	
	unsigned lc = 127;
	lc = lc & (0xffffffff - 31);
	printf("32 bit align lc %d\n", lc);
	printf("32 bit align lc %s\n", byte_to_binary(lc));
	printf("2199              %s\n", byte_to_binary(2199));

	printf("31              %s\n", byte_to_binary(31));
	
	printf("2199 & 31         %s\n", byte_to_binary(2199 & 31));
	printf("2199 & 31         %d\n", 2199 & 31);
	
	printf("node sz %d\n", (int)sizeof(KdTreeNode));
	printf("prim sz %d\n", (int)sizeof(Primitive));
	printf("event sz %d\n", (int)sizeof(SplitEvent));
	printf("bbox sz %d\n", (int)sizeof(BoundingBox));*/
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
	printf("tree bbox: %f %f %f - %f %f %f\n", m_bbox.min(0), m_bbox.min(1), m_bbox.min(2), m_bbox.max(0), m_bbox.max(1), m_bbox.max(2));
	
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
	if(ctx.getNumPrimitives() < 32 || level == 22) {
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
		
	delete leftCtx;

	if(plane->rightCount() > 0)
		subdivide(branch + 1, *rightCtx, level + 1);
		
	delete rightCtx;
}

char KdTree::intersect(const Ray &ray, RayIntersectionContext & ctx)
{
	float hitt0, hitt1;
	if(!m_bbox.intersect(ray, &hitt0, &hitt1)) return 0;
	
	ctx.setBBox(m_bbox);

	KdTreeNode * root = getRoot();
	return recusiveIntersect(root, ray, ctx);
}

char KdTree::recusiveIntersect(KdTreeNode *node, const Ray &ray, RayIntersectionContext & ctx)
{
	printf("recus intersect level %i\n", ctx.m_level);
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
	BoundingBox bigBox = ctx.getBBox();
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
		ctx.setBBox(nearBox);
		ctx.m_level++;
		if(recusiveIntersect(nearNode, ray, ctx)) return 1;
	
		if(tplane < ray.m_tmin || tplane > ray.m_tmax)
			return 0;
		
		ctx.setBBox(farBox);
		ctx.m_level--;
		if(recusiveIntersect(farNode, ray, ctx)) return 1;
	}
	else {
		if(tplane > 0) {
			float hitt0, hitt1;
			bigBox.intersect(ray, &hitt0, &hitt1);
			if(tplane > hitt1) {
				ctx.setBBox(nearBox);
				ctx.m_level++;
				if(recusiveIntersect(nearNode, ray, ctx)) return 1;
			}
			else {
				ctx.setBBox(farBox);
				ctx.m_level++;
				if(recusiveIntersect(farNode, ray, ctx)) return 1;
			}
		}
		else {
				ctx.setBBox(nearBox);
				ctx.m_level++;
				if(recusiveIntersect(nearNode, ray, ctx)) return 1;
		}
	}
	return 0;
}

char KdTree::leafIntersect(KdTreeNode *node, const Ray &ray, RayIntersectionContext & ctx)
{
	unsigned start = node->getPrimStart();
	unsigned num = node->getNumPrims();
	
	printf("prims count in leaf %i start at %i\n", node->getNumPrims(), node->getPrimStart());
	IndexArray &indir = m_stream.indirection();
	PrimitiveArray &prims = m_stream.primitives();
	indir.setIndex(start);
	for(unsigned i = 0; i < num; i++) {
		unsigned *iprim = indir.asIndex();

		Primitive * prim = prims.asPrimitive(*iprim);
		BaseMesh *mesh = (BaseMesh *)prim->getGeometry();
		unsigned iface = prim->getComponentIndex();
		
		//printf("i prim %i i face %i", *iprim, iface);
		mesh->intersect(iface, ray, ctx);
		indir.next();
	}
	if(ctx.m_success) printf("hit");
	return ctx.m_success;
}

