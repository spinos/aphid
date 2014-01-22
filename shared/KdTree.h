/*
 *  KdTree.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Geometry.h>
#include <KdTreeNode.h>
#include <BuildKdTreeStream.h>
#include <KdTreeBuilder.h>

class IntersectionContext;
class SelectionContext;
class KdTree : public Geometry
{
public:
	KdTree();
	~KdTree();
	
	bool isEmpty() const;
	KdTreeNode* getRoot() const;
	void cleanup();
	void addMesh(BaseMesh* mesh);
	void create();
	
	char intersect(IntersectionContext * ctx);
	char closestPoint(const Vector3F & origin, IntersectionContext * ctx);
	void select(SelectionContext * ctx);

	Primitive * getPrim(unsigned idx);
	
private:
	void subdivide(KdTreeNode * node, BuildKdTreeContext & ctx, int level);
	void createLeaf(KdTreeNode * node, BuildKdTreeContext & ctx);
	char recusiveIntersect(KdTreeNode *node, IntersectionContext * ctx);
	char leafIntersect(KdTreeNode *node, IntersectionContext * ctx);
	char recusiveClosestPoint(KdTreeNode *node, const Vector3F &origin, IntersectionContext * ctx);
	char leafClosestPoint(KdTreeNode *node, const Vector3F &origin, IntersectionContext * ctx);
	char recursiveSelect(KdTreeNode *node, SelectionContext * ctx);
	char leafSelect(KdTreeNode *node, SelectionContext * ctx);
	BuildKdTreeStream m_stream;
	KdTreeNode *m_root;
	int m_maxLeafLevel;
};