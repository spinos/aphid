/*
 *  KdTree.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <KdTreeNode.h>
#include <BaseMesh.h>
#include <BoundingBox.h>
#include <Primitive.h>
#include <BuildKdTreeStream.h>
#include <KdTreeBuilder.h>

class IntersectionContext;
typedef Primitive * primitivePtr;
	
class KdTree
{
public:
	KdTree();
	~KdTree();
	
	KdTreeNode* getRoot() const;
	void cleanup();
	void addMesh(BaseMesh* mesh);
	void create();
	
	char intersect(const Ray &ray, IntersectionContext * ctx);
	char closestPoint(const Vector3F & origin, IntersectionContext * ctx);

	BoundingBox m_bbox;
	
	Primitive * getPrim(unsigned idx);
	
private:
	void subdivide(KdTreeNode * node, BuildKdTreeContext & ctx, int level);
	char recusiveIntersect(KdTreeNode *node, const Ray &ray, IntersectionContext * ctx);
	char leafIntersect(KdTreeNode *node, const Ray &ray, IntersectionContext * ctx);
	char recusiveClosestPoint(KdTreeNode *node, const Vector3F &origin, IntersectionContext * ctx);
	char leafClosestPoint(KdTreeNode *node, const Vector3F &origin, IntersectionContext * ctx);
	
	BuildKdTreeStream m_stream;
	KdTreeNode *m_root;
};