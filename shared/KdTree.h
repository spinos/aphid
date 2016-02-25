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
#include <Boundary.h>
#include <KdTreeNode.h>
#include <BuildKdTreeStream.h>
#include <KdTreeBuilder.h>
#include <VectorArray.h>

namespace aphid {

class IntersectionContext;
class SelectionContext;
class KdTree : public Geometry, public Boundary
{
	int m_minNumLeafPrims, m_maxNumLeafPrims;
	BoundingBox m_testBox;
	unsigned m_intersectElement;
	std::string m_buildLogStr;
    
public:
	KdTree();
	virtual ~KdTree();
	
	bool isEmpty() const;
	KdTreeNode* getRoot() const;
	void addGeometry(Geometry * geo);
	
	virtual void create();
	
	char intersect(IntersectionContext * ctx);
	void select(SelectionContext * ctx);

	Primitive * getPrim(unsigned idx);
	virtual const Type type() const;
	static int MaxBuildLevel;
	static unsigned NumPrimitivesInLeafThreashold;
// override geomery
	virtual bool intersectBox(const BoundingBox & box);
	virtual void closestToPoint(ClosestToPointTestResult * result);
	std::string buildLog() const;
    
protected:
	BuildKdTreeStream m_stream;
	
protected:
	virtual void clear();
	const unsigned numNoEmptyLeaves() const;
	sdb::VectorArray<Primitive> & indirection();
	
private:
	void subdivide(KdTreeNode * node, BuildKdTreeContext & ctx, int level);
	void createLeaf(KdTreeNode * node, BuildKdTreeContext & ctx);
	char recusiveIntersect(KdTreeNode *node, IntersectionContext * ctx);
	char leafIntersect(KdTreeNode *node, IntersectionContext * ctx);
	char recursiveSelect(KdTreeNode *node, SelectionContext * ctx);
	char leafSelect(KdTreeNode *node, SelectionContext * ctx);
	void recusiveClosestToPoint(KdTreeNode *node, const BoundingBox &box, ClosestToPointTestResult * result);
	void leafClosestToPoint(KdTreeNode *node, const BoundingBox &box, ClosestToPointTestResult * result);
	bool recursiveIntersectBox(KdTreeNode *node, const BoundingBox & box);
	bool leafIntersectBox(KdTreeNode *node, const BoundingBox & box);
	
	KdTreeNode *m_root;
	int m_maxLeafLevel;
	unsigned m_numNoEmptyLeaf;
};

}