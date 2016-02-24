/*
 *  KdCluster.cpp
 *  testkdtree
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "KdCluster.h"
#include <GeometryArray.h>

namespace aphid {

KdCluster::KdCluster() : m_groupGeometries(0) {}
KdCluster::~KdCluster() {}

const unsigned KdCluster::numGroups() const
{ return numNoEmptyLeaves(); }

GeometryArray * KdCluster::group(unsigned idx) const
{ return m_groupGeometries[idx]; }

void KdCluster::setGroupGeometry(unsigned idx, GeometryArray * geos)
{ m_groupGeometries[idx] = geos; }

void KdCluster::create()
{
	KdTree::create();
	KdTreeNode * root = getRoot();
	if(!root) return;
	
	m_groupGeometries = new GeometryArray *[numNoEmptyLeaves()];
	
	m_nodeGroupInd.clear();
	m_currentGroup = 0;
	const BoundingBox b = getBBox();
	recursiveFindGroup(root, b);
	
	unsigned totalNGeo = 0;
	unsigned minN = 1<<20;
	unsigned maxN = 0;
	unsigned n;
	unsigned i=0;
	for(; i < m_currentGroup; i++) {
		n = m_groupGeometries[i]->numGeometries();
		totalNGeo += n;
		if(n < minN) minN = n;
		if(n > maxN) maxN = n;
	}
	
	std::cout<<" total n geometries "<<totalNGeo<<"\n"
	<<" kd clustering to n groups "<<m_currentGroup<<"\n"
	<<" min/max geometry count in group "<<minN<<"/"<<maxN<<"\n";
}

void KdCluster::clear()
{
	clearGroups();
	KdTree::clear();
}

void KdCluster::clearGroups()
{
	if(!m_groupGeometries) return;
	delete[] m_groupGeometries;
	m_groupGeometries = 0;
}

void KdCluster::recursiveFindGroup(KdTreeNode *node, const BoundingBox & box)
{
	if(node->isLeaf()) {
		leafWriteGroup(node, box);
		return;
	}
	
	const int axis = node->getAxis();
	const float splitPos = node->getSplitPos();
	
	BoundingBox leftBox, rightBox;
	box.split(axis, splitPos, leftBox, rightBox);
	
	recursiveFindGroup(node->getLeft(), leftBox);
	recursiveFindGroup(node->getRight(), rightBox);
}

void KdCluster::leafWriteGroup(KdTreeNode *node, const BoundingBox & box)
{
	const unsigned num = node->getNumPrims();
	if(num < 1) return;
	
	m_groupGeometries[m_currentGroup] = new GeometryArray;
	
	GeometryArray * curGrp = m_groupGeometries[m_currentGroup];
	curGrp->create(num);
	
	unsigned start = node->getPrimStart();
	sdb::VectorArray<Primitive> &indir = indirection();
	//sdb::VectorArray<Primitive> &prims = primitives();
	int igeom, icomponent;
	unsigned igroup = 0;
	for(unsigned i = 0; i < num; i++) {
		//unsigned *iprim = indir[start + i];
		//Primitive * prim = prims.get(*iprim);
		Primitive * prim = indir[start + i];
		prim->getGeometryComponent(igeom, icomponent);
		Geometry * geo = m_stream.geometry(igeom);
		if(geo->type() == TGeometryArray) {
			GeometryArray * ga = (GeometryArray *)geo;
			Geometry * comp = ga->geometry(icomponent);
			
			BoundingBox comb = ga->calculateBBox(icomponent);

// do not add straddling geo			
			if(comb.getMax(0) <= box.getMax(0) && 
				comb.getMax(1) <= box.getMax(1) && 
				comb.getMax(2) <= box.getMax(2)) 
			{
				curGrp->setGeometry(comp, igroup);
				igroup++;
			}
		}
		else {
			std::cout<<" grouping only works with geometry arry.";
		}
		//indir.next();
	}
	
	curGrp->setNumGeometries(igroup);
	m_nodeGroupInd[node] = m_currentGroup;
	m_currentGroup++;
}

bool KdCluster::intersectRay(const Ray * eyeRay)
{ 
	if(!getRoot()) return false;
	float hitt0, hitt1;
	BoundingBox b = getBBox();
	if(!b.intersect(*eyeRay, &hitt0, &hitt1)) return false;
	
	return recursiveIntersectRay(getRoot(), eyeRay, b); 
}

bool KdCluster::recursiveIntersectRay(KdTreeNode *node, const Ray * eyeRay, const BoundingBox & box)
{
	if(node->isLeaf())
		return leafIntersectRay(node, eyeRay);
		
	const int axis = node->getAxis();
	const float splitPos = node->getSplitPos();
	
	BoundingBox leftBox, rightBox;
	box.split(axis, splitPos, leftBox, rightBox);
	
	float lambda1, lambda2;
    float mu1, mu2;
	char b1 = leftBox.intersect(*eyeRay, &lambda1, &lambda2);
	char b2 = rightBox.intersect(*eyeRay, &mu1, &mu2);
	
	if(b1 && b2) {
		if(mu1 < lambda1) {
// vist right child first
			if(recursiveIntersectRay(node->getRight(), eyeRay, rightBox)) return true;
			if(recursiveIntersectRay(node->getLeft(), eyeRay, leftBox)) return true;
		}
		else {
// vist left child first
			if(recursiveIntersectRay(node->getLeft(), eyeRay, leftBox)) return true;
			if(recursiveIntersectRay(node->getRight(), eyeRay, rightBox)) return true;
		}
	}
	else if(b1) {
		if(recursiveIntersectRay(node->getLeft(), eyeRay, leftBox)) return true;
	}
	else if(b2) {
		if(recursiveIntersectRay(node->getRight(), eyeRay, rightBox)) return true;
	}
	return false;
}

bool KdCluster::leafIntersectRay(KdTreeNode *node, const Ray * eyeRay)
{
	const unsigned num = node->getNumPrims();
	if(num < 1) return false;
	
	GeometryArray * g = group(m_nodeGroupInd[node]);
	if(!g->intersectRay(eyeRay)) return false;
	m_currentGroup = m_nodeGroupInd[node];
	return true;
}

const unsigned KdCluster::currentGroup() const
{ return m_currentGroup; }

void KdCluster::setCurrentGroup(unsigned x)
{ m_currentGroup = x; }

bool KdCluster::isGroupIdValid(unsigned x) const
{ return x < numGroups(); }

}
//:~