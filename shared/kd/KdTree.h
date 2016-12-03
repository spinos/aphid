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
#include <kd/KdTreeNode.h>
#include <kd/BuildKdTreeStream.h>
#include <kd/KdTreeBuilder.h>
#include <sdb/VectorArray.h>
#include <kd/TreeProperty.h>

namespace aphid {

class IntersectionContext;
class SelectionContext;
class KdTree : public Geometry, public Boundary, public TreeProperty
{
	BoundingBox m_testBox;
	unsigned m_intersectElement;
	std::string m_buildLogStr;
    
public:
	KdTree();
	virtual ~KdTree();
	
	bool isEmpty() const;
	KdTreeNode* getRoot() const;
	void addGeometry(Geometry * geo);
	
	virtual void create(BuildProfile * prof);
	
	char intersect(IntersectionContext * ctx);
	void select(SelectionContext * ctx);

	Primitive * getPrim(unsigned idx);
	virtual const Type type() const;
	
// override geomery
	virtual bool intersectBox(const BoundingBox & box);
	virtual void closestToPoint(ClosestToPointTestResult * result);
	std::string buildLog() const;
    
protected:
	BuildKdTreeStream m_stream;
	
protected:
	virtual void clear();
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
	
};

}