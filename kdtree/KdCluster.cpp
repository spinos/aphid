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
KdCluster::KdCluster() : m_groupGeometries(0) {}
KdCluster::~KdCluster() {}

const unsigned KdCluster::numGroups() const
{ return numNoEmptyLeaves(); }

GeometryArray * KdCluster::group(unsigned idx) const
{ return m_groupGeometries[idx]; }

void KdCluster::create()
{
	KdTree::create();
	KdTreeNode * root = getRoot();
	if(!root) return;
	
	m_groupGeometries = new GeometryArray *[numNoEmptyLeaves()];
	
	m_currentGroup = 0;
	const BoundingBox b = getBBox();
	recursiveFindGroup(root, b);
	
	unsigned totalNGeo = 0;
	unsigned minN = 1<<20;
	unsigned maxN = 0;
	unsigned n;
	unsigned i=0;
	for(; i < m_currentGroup; i++) {
		n = m_groupGeometries[i]->numGeometies();
		totalNGeo += n;
		if(n < minN) minN = n;
		if(n > maxN) maxN = n;
	}
	
	std::cout<<" total n geometries "<<totalNGeo<<"\n"
	<<" kd clustering to n groups "<<m_currentGroup<<"\n"
	<<" min/max geometry count in group "<<minN<<"/"<<maxN<<"\n";
}

void KdCluster::rebuild()
{
	KdTree::rebuild();
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
	IndexArray &indir = indirection();
	PrimitiveArray &prims = primitives();
	indir.setIndex(start);

	unsigned igroup = 0;
	for(unsigned i = 0; i < num; i++) {
		unsigned *iprim = indir.asIndex();

		Primitive * prim = prims.asPrimitive(*iprim);
		Geometry * geo = prim->getGeometry();
		unsigned icomponent = prim->getComponentIndex();
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
			return;
		}
		indir.next();
	}
	
	curGrp->setNumGeometries(igroup);
	m_currentGroup++;
}
