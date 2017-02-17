/*
 *  LodGrid.cpp
 *  
 *
 *  Created by jian zhang on 2/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "LodGrid.h"
#include <math/miscfuncs.h>

namespace aphid {

namespace sdb {

LodNode::LodNode()
{}

LodNode::~LodNode()
{}

LodCell::LodCell(Entity * parent) :
Array<int, LodNode >(parent)
{}

LodCell::~LodCell()
{}

void LodCell::clear()
{ 
	TParent::clear();
}

void LodCell::countNodesInCell(int & it)
{
	begin();
	while(!end() ) {
		value()->index = it;
		it++;

		next();
	}
}

void LodCell::dumpNodesInCell(LodNode * dst)
{
	begin();
	while(!end() ) {
		LodNode * a = value();
		dst[a->index] = *a;
		
		next();
	}
}

LodGrid::LodGrid(Entity * parent) : TParent(parent)
{}

LodGrid::~LodGrid()
{}

void LodGrid::resetBox(const BoundingBox & b,
				const float & h)
{
	clear();
	setLevel0CellSize(h);
	const Coord4 lc = cellCoordAtLevel(b.center(), 0);
	addCell(lc);
	calculateBBox();
}

void LodGrid::fillBox(const BoundingBox & b,
				const float & h)
{
	clear();
	setLevel0CellSize(h);
	
	const int s = level0CoordStride();
	const Coord4 lc = cellCoordAtLevel(b.getMin(), 0);
	const Coord4 hc = cellCoordAtLevel(b.getMax(), 0);
	const int dimx = (hc.x - lc.x) / s + 1;
	const int dimy = (hc.y - lc.y) / s + 1;
	const int dimz = (hc.z - lc.z) / s + 1;
	const float fh = finestCellSize();
	
	const Vector3F ori(fh * (lc.x + s/2),
						fh * (lc.y + s/2),
						fh * (lc.z + s/2));
						
	int i, j, k;
	Coord4 sc;
	sc.w = 0;
	for(k=0; k<dimz;++k) {
		sc.z = lc.z + s * k;
			for(j=0; j<dimy;++j) {
			sc.y = lc.y + s * j;
			for(i=0; i<dimx;++i) {
				sc.x = lc.x + s * i;
				LodCell * cell = findCell(sc);
				if(!cell) { 
					addCell(sc);
				}
				
			}
		}
	}
	
	calculateBBox();
}

void LodGrid::clear()
{
	TParent::clear(); 
}

int LodGrid::countLevelNodes(int level)
{
	int c = 0;
	begin();
	while(!end() ) {
		if(key().w == level) {
			value()->countNodesInCell(c);
		}
		
		if(key().w > level) {
			break;
		}

		next();
	}
	
	return c;
}

void LodGrid::dumpLevelNodes(LodNode * dst, int level)
{
	begin();
	while(!end() ) {
		if(key().w == level) {
			value()->dumpNodesInCell(dst);
		}
		
		if(key().w > level) {
			break;
		}

		next();
	}
}

void LodGrid::inserNodedByAggregation(int minLevel, int maxLevel)
{
	for(int i = maxLevel;i>=minLevel;--i) {
		aggregateAtLevel(i);
	}
}

void LodGrid::aggregateAtLevel(int level)
{
	begin();
	while(!end() ) {
		if(key().w == level) {
			
			LodCell * cell = value();
			if(cell->hasChild() ) {
				aggregateInCell(cell, key() );
			}
		}
		
		if(key().w > level) {
			break;
		}

		next();
	}
}

void LodGrid::aggregateInCell(LodCell * cell, 
						const Coord4 & cellCoord)
{
	int n = 0;
	for(int i=0; i< 8; ++i) { 
		const Coord4 childC = childCoord(cellCoord, i);
		
		LodCell * childCell = findCell(childC);
		if(childCell) {
			childCell->countNodesInCell(n);
		}
	}
	
	LodNode * nds = new LodNode[n];
	for(int i=0; i< 8; ++i) { 
		const Coord4 childC = childCoord(cellCoord, i);
		
		LodCell * childCell = findCell(childC);
		if(childCell) {
			childCell->dumpNodesInCell(nds);
		}
	}
	
	if(n > 12) {
		const float sepd = levelCellSize(cellCoord.w + 3) * .1f;
		processKmean(n, nds, sepd);
	}
	
	for(int i=0; i< n; ++i) { 
		LodNode * par = new LodNode;
		*par = nds[i];
		par->index = -1;
					
		cell->insert(i, par);
	}
	
	delete[] nds;
	
}

void LodGrid::processKmean(int & n, 
					LodNode * samples,
					const float & separateDist)
{
	int k = n - 2;
	if(n > 24) {
		k = n - n / 3;
	}
	if(n > 48) {
		k = n - n / 2;
	}
	
/// position and normal
	const int d = 6;
/// to kmean data
	DenseMatrix<float> data(n, d);
	for(int i=0;i<n;++i) {
		const LodNode & src = samples[i];
		data.column(0)[i] = src.pos.x;
		data.column(1)[i] = src.pos.y;
		data.column(2)[i] = src.pos.z;
		data.column(3)[i] = src.nml.x * 2.5f;
		data.column(4)[i] = src.nml.y * 2.5f;
		data.column(5)[i] = src.nml.z * 2.5f;
	}
	
	KMeansClustering2<float> cluster;
	cluster.setKND(k, n, d);
	cluster.setSeparateDistance(separateDist);
	if(!cluster.compute(data) ) {
		std::cout<<"\n LodGrid kmean failed ";
		return;
	}
/// from kmean data	
	DenseVector<float> centr;
	for(int i=0;i<k;++i) {
		cluster.getGroupCentroid(centr, i);
		LodNode & dst = samples[i];
		dst.pos.set(centr[0], centr[1], centr[2]);
		dst.nml.set(centr[3], centr[4], centr[5]);
		dst.nml.normalize();
		
	}
	n = k;
}

}

}
