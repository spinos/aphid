/*
 *  KdIntersection.cpp
 *  testkdtree
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "KdIntersection.h"
#include <GeometryArray.h>
KdIntersection::KdIntersection() {}
KdIntersection::~KdIntersection() {}

bool KdIntersection::intersectBox(const BoundingBox & box)
{
	KdTreeNode * root = getRoot();
	if(!root) return false;
	
	BoundingBox b = getBBox();
	
	m_testBox = box;
	
	return recursiveIntersectBox(root, b);
}

bool KdIntersection::recursiveIntersectBox(KdTreeNode *node, const BoundingBox & box)
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

bool KdIntersection::leafIntersectBox(KdTreeNode *node, const BoundingBox & box)
{
	const unsigned num = node->getNumPrims();
	if(num < 1) return false;
	
	unsigned start = node->getPrimStart();
	IndexArray &indir = indirection();
	PrimitiveArray &prims = primitives();
	indir.setIndex(start);

	for(unsigned i = 0; i < num; i++) {
		unsigned *iprim = indir.asIndex();

		Primitive * prim = prims.asPrimitive(*iprim);
		Geometry * geo = prim->getGeometry();
		unsigned icomponent = prim->getComponentIndex();
		
		if(geo->intersectBox(icomponent, m_testBox)) {
			m_intersectElement = icomponent;
			return true;
		}
		indir.next();
	}
	return false;
}

bool KdIntersection::intersectTetrahedron(const Vector3F * tet)
{ 
	KdTreeNode * root = getRoot();
	if(!root) return false;
	
	m_testTetrahedron[0] = tet[0];
	m_testTetrahedron[1] = tet[1];
	m_testTetrahedron[2] = tet[2];
	m_testTetrahedron[3] = tet[3];
	
	m_testBox.reset();
	m_testBox.expandBy(tet[0]);
	m_testBox.expandBy(tet[1]);
	m_testBox.expandBy(tet[2]);
	m_testBox.expandBy(tet[3]);
	
	BoundingBox b = getBBox();
	
	return recursiveIntersectTetrahedron(root, b); 
}

bool KdIntersection::recursiveIntersectTetrahedron(KdTreeNode *node, const BoundingBox & box)
{
	if(node->isLeaf())
		return leafIntersectTetrahedron(node, box);
		
	if(!box.intersect(m_testBox)) return false;
	
	const int axis = node->getAxis();
	const float splitPos = node->getSplitPos();
	
	BoundingBox leftBox, rightBox;
	box.split(axis, splitPos, leftBox, rightBox);
	
	if(recursiveIntersectTetrahedron(node->getLeft(), leftBox)) return true;
	
	return recursiveIntersectTetrahedron(node->getRight(), rightBox);
}

bool KdIntersection::leafIntersectTetrahedron(KdTreeNode *node, const BoundingBox & box)
{
	const unsigned num = node->getNumPrims();
	if(num < 1) return false;
	
	if(!box.intersect(m_testBox)) return false;
	
	unsigned start = node->getPrimStart();
	IndexArray &indir = indirection();
	PrimitiveArray &prims = primitives();
	indir.setIndex(start);

	for(unsigned i = 0; i < num; i++) {
		unsigned *iprim = indir.asIndex();

		Primitive * prim = prims.asPrimitive(*iprim);
		Geometry * geo = prim->getGeometry();
		unsigned icomponent = prim->getComponentIndex();
		
		if(geo->intersectTetrahedron(icomponent, m_testTetrahedron)) {
			m_intersectElement = icomponent;
			return true;
		}
		indir.next();
	}
	return false;
}

unsigned KdIntersection::countElementIntersectBox(std::vector<unsigned> & result, const BoundingBox & box)
{
    KdTreeNode * root = getRoot();
	if(!root) return 0;
	
	BoundingBox b = getBBox();
	
	m_testBox = box;
	
    result.clear();
	internalCountElementIntersectBox(result, root, b);
    return result.size();
}

void KdIntersection::internalCountElementIntersectBox(std::vector<unsigned> & result, KdTreeNode *node, const BoundingBox & box)
{
    if(!box.intersect(m_testBox)) return;
	
	if(node->isLeaf()) return leafCountElementIntersectBox(result, node, box);
		
	const int axis = node->getAxis();
	const float splitPos = node->getSplitPos();
	
	BoundingBox leftBox, rightBox;
	box.split(axis, splitPos, leftBox, rightBox);
	
	internalCountElementIntersectBox(result, node->getLeft(), leftBox);
	internalCountElementIntersectBox(result, node->getRight(), rightBox);
}

void KdIntersection::leafCountElementIntersectBox(std::vector<unsigned> & result, KdTreeNode *node, const BoundingBox & box)
{
    const unsigned num = node->getNumPrims();
	if(num < 1) return;
	
	unsigned start = node->getPrimStart();
	IndexArray &indir = indirection();
	PrimitiveArray &prims = primitives();
	indir.setIndex(start);

	for(unsigned i = 0; i < num; i++) {
		unsigned *iprim = indir.asIndex();

		Primitive * prim = prims.asPrimitive(*iprim);
		Geometry * geo = prim->getGeometry();
		unsigned icomponent = prim->getComponentIndex();
		
		if(geo->intersectBox(icomponent, m_testBox)) result.push_back(icomponent);
		indir.next();
	}
}

unsigned KdIntersection::intersectedElement() const
{ return m_intersectElement; }
//:~