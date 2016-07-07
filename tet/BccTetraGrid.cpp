/*
 *  BccTetraGrid.cpp
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <BccTetraGrid.h>

using namespace aphid;

namespace ttg {

BccTetraGrid::BccTetraGrid() 
{}

BccTetraGrid::~BccTetraGrid() 
{}

void BccTetraGrid::addBlueNodes(const Vector3F & cellCenter)
{
	BccCell c(cellCenter);
	c.addRedBlueNodes(this, gridCoord((const float *)c.centerP() ) );
}

void BccTetraGrid::buildNodes()
{
	std::vector<BccCell> cells;
	begin();
	while(!end() ) {
		cells.push_back(BccCell(coordToCellCenter(key() ) ) );
		next();
	}
	
	const int n = cells.size();
	int i=0;
	for(;i<n;++i) {
		const BccCell & c = cells[i];
		c.addRedBlueNodes(this, gridCoord((const float *)c.centerP() ) );
	}
	cells.clear();
	countNodes();
	
}

/// sequenc node index
void BccTetraGrid::countNodes()
{
	int c = 0;
	begin();
	while(!end() ) {
		countNodesIn(value(), c );
		next();
	}
}

void BccTetraGrid::countNodesIn(aphid::sdb::Array<int, BccNode> * cell, int & c)
{
	cell->begin();
	while(!cell->end() ) {
#if 0
		if(cell->key() > 14999) 
			std::cout<<"\n node"<<cell->key()<<" "<<c
		<<" "<<cell->value()->prop;
#endif
		cell->value()->index = c;
		c++;
		cell->next();
	}
}

int BccTetraGrid::numNodes()
{
	int c = 0;
	begin();
	while(!end() ) {
		c+= value()->size();
		next();
	}
	return c;
}

void BccTetraGrid::extractNodePosProp(Vector3F * destPos,
					int * destProp)
{
	begin();
	while(!end() ) {
		
		extractNodePosPropIn(destPos, destProp, value() );
		next();
	}
}

void BccTetraGrid::extractNodePosPropIn(Vector3F * destPos,
					int * destProp,
					sdb::Array<int, BccNode> * cell)
{
	cell->begin();
	while(!cell->end() ) {
		int i = cell->value()->index;
		destPos[i] = cell->value()->pos;
		destProp[i] = cell->value()->prop;
		
		cell->next();
	}
}

void BccTetraGrid::buildTetrahedrons(std::vector<ITetrahedron *> & dest)
{
	countNodes();
	
	std::vector<BccCell> cells;
	begin();
	while(!end() ) {
		cells.push_back(BccCell(coordToCellCenter(key() ) ) );
		next();
	}
	
	STriangleArray faces;
	
	const int n = cells.size();
	int i=0;
	for(;i<n;++i) {
		const BccCell & c = cells[i];
		c.connectNodes(dest, this, gridCoord((const float *)c.centerP() ), &faces );
	}
	cells.clear();
	std::cout<<"\n n face "<<faces.size();
	std::cout.flush();
	
	faces.begin();
	while(!faces.end() ) {
		STriangle<ITetrahedron> * f = faces.value();
		if(f->tb) {
			bool stat = connectTetrahedrons(f->ta, f->tb,
								f->key.x, f->key.y, f->key.z);
			if(!stat) {
				printTetrahedronCannotConnect(f->ta, f->tb);
			}
		}
		faces.next();
	}
	faces.clear();
}

void BccTetraGrid::moveNodeIn(const aphid::Vector3F & cellCenter,
					const Vector3F * pos, 
					const int & n,
					aphid::Vector3F * X,
					int * prop)
{
/// each node can move once
	int moved[16];
	int i = 0;
	for(;i<16;++i)
		moved[i] = 0;
		
	BccCell cell(cellCenter );
	
	Vector3F closestP;
	float minD = 1e8f;
	float d;
	for(i=0;i<n;++i) {
		d = cellCenter.distanceTo(pos[i]);
		if(d<minD) {
			minD = d;
			closestP = pos[i];
		}
	}
	
	int xi;
#if 1
	if(cell.moveNode(xi, this, gridCoord((const float *)&cellCenter ),
					closestP,
					moved) ) {
		X[xi] = closestP;
		prop[xi] = 1;
	}
#else
	for(i=0;i<n;++i) {
		if(cell.moveNode(xi, this, gridCoord((const float *)&cellCenter ),
					pos[i],
					moved) ) {
			X[xi] = pos[i];
			prop[xi] = 1;
		}
	}
#endif
}

bool BccTetraGrid::moveRedNodeTo(const Vector3F & cellCenter,
					const sdb::Coord3 & cellCoord,
					const Vector3F & pos)
{	
/// guarantee in which red-blue distance
	const float r = gridSize() * .13f;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	
	Vector3F dp, blueP;
/// for each blue
	int i=0;
	for(;i<8;++i) {
		BccNode * xi = fCell.blueNode(i, cell, this, cellCoord, blueP);
		
		dp = pos - blueP;
		if(Absolute<float>(dp.x) < r) return false;
		if(Absolute<float>(dp.y) < r) return false;
		if(Absolute<float>(dp.z) < r) return false;	
		
	}
	
	BccNode * node = fCell.redNode(this, cellCoord );
	
	// std::cout<<"\n move red";
	node->pos = pos;
	node->prop = 4;
	
	return true;
}

void BccTetraGrid::moveBlueNodes(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{
	const float gz = gridSize();
	const float r = gz * .25f;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	
	float d;
	Vector3F blueP, closestP;
/// for each blue
	int i=0;
	for(;i<8;++i) {
		BccNode * xi = fCell.blueNode(i, cell, this, cellCoord, blueP);
/// already moved
		if(xi->prop > 0) 
			continue;
			
		samples->getClosest(closestP, d, blueP);
		
		if(!fCell.moveBlueTo(closestP, blueP, r) )
			continue;
			
		// std::cout<<"\n move blue";
		xi->pos = closestP;
		xi->prop = 3;
	}
}

Vector3F BccTetraGrid::moveRedToCellCenter(const Vector3F & cellCenter,
					const sdb::Coord3 & cellCoord)
{
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	Vector3F blueP;
	Vector3F average(0.f, 0.f, 0.f);
	int i=0;
	for(;i<8;++i) {
		BccNode * xi = fCell.blueNode(i, cell, this, cellCoord, blueP);
		average += blueP;
	}
	average *= .125;
	BccNode * ni = fCell.redNode(this, cellCoord);
	ni->pos = average;
	return average;
}

void BccTetraGrid::cutRedRedEdges(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{
	const float gz = gridSize();
	const float r = gz * .25f;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	BccNode * redn = fCell.redNode(this, cellCoord);
	Vector3F redP = redn->pos;
	Vector3F antiRedP;
	bool redOnFront;
	int blueOnFront;
	float d;
	Vector3F q, closestP;
	Vector3F facePs[8];
	int nfaceP;
/// for each red-red
	int i=0;
	for(;i<6;++i) {
		bool toAdd = true;
/// already cut
		if(fCell.redRedNode(i, cell, this, cellCoord) )
			toAdd = false;
		
		if(toAdd) {
			fCell.facePosition(q, facePs, nfaceP, blueOnFront, i, cell, this, cellCoord);
			BccNode * nend = fCell.faceNode(i, cell, this, cellCoord);
			redOnFront = (redn->prop > -1 && nend->prop > -1);
			antiRedP = nend->pos;
			
			if(blueOnFront<1 && !redOnFront) 
				toAdd = fCell.checkWedgeFace(redP, antiRedP, facePs, r);
		}
			
		if(toAdd) {
/// cut add face center
			BccNode * node = fCell.addFaceNode(i, this, cellCoord);
			node->pos = q;
			
/// mid pnt on front
			if(redOnFront || blueOnFront>3) 
				node->prop = 5;
			else
				node->prop = -1;
			
/// move to sample if possible
			samples->getClosest(closestP, d, q);
			
			if(fCell.checkSplitFace(closestP, redP, antiRedP, r, i/2, facePs, nfaceP) ) {
				node->pos = closestP;
				node->prop = 5;
			}
		}
		
	}
}

void BccTetraGrid::cutBlueBlueEdges(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{
	const float gz = gridSize();
	const float r = gz * .25f;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	
	Vector3F blueP[8];
	int blueProp[8];
	int i=0;
	for(;i<8;++i) {
		BccNode * node = fCell.blueNode(i, cell, this, cellCoord, blueP[i]);
		blueProp[i] = node->prop;
	}
	
	int v1, v2, prop1, prop2;
	Vector3F p1, p2, q, closestP;
	float d;

	i=0;
	for(;i<12;++i) {
/// already cut
		if(fCell.blueBlueNode(i, cell, this, cellCoord) )
			continue;
			
		fCell.blueBlueEdgeV(v1, v2, i);
		p1 = blueP[v1]; p2 = blueP[v2];
		prop1 = blueProp[v1]; prop2 = blueProp[v2];
		
		q = (p1 + p2) * .5f;
		samples->getClosest(closestP, d, q) > -1;
		
/// skip edge on front
		bool toAdd = prop1 < 0 || prop2 < 0;
		
		if(toAdd) {
			toAdd = fCell.checkSplitEdge(closestP, p1, p2, r, i/4);
		}
		
		if(toAdd) {
			BccNode * node = fCell.addEdgeNode(i, this, cellCoord);
			node->pos = closestP;
			node->prop = 6;
		}
	}
}

void BccTetraGrid::loopBlueBlueEdges(const Vector3F & cellCenter,
					const sdb::Coord3 & cellCoord)
{
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	Vector3F p1, p2;
	bool toLoop = false;
/// for each face
	int i = 0, j;
	for(;i<6;++i) {
		if(fCell.anyBlueCut(i, cell, this, cellCoord) ) {
			toLoop = true;
			break;
		}
	}
	
	if(!toLoop)
		return;
		
	for(i=0;i<6;++i) {
/// for each blue-blue edge			
		for(j=0;j<4;++j) {
			
			BccNode * nodeC = fCell.faceVaryBlueBlueNode(i, j, cell, this, cellCoord);
/// alread cut
			if(nodeC)
				continue;
				
			fCell.faceEdgePostion(p1, p2, i, j, cell, this, cellCoord);
			
			nodeC = fCell.addFaceVaryEdgeNode(i, j, this, cellCoord);
			nodeC->pos = (p1 + p2) * .5f;
		}
	}
}

void BccTetraGrid::cutRedBlueEdges(const Vector3F & cellCenter,
					const sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{
	return;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	
	BccNode * redN = fCell.redNode(this, cellCoord);
	if(redN->prop > 0)
		return;
	const Vector3F redP = redN->pos;
	
	Vector3F blueP;
/// for each blue 
	int i=0;
	for(;i<8;++i) {
		BccNode * blueN = fCell.blueNode(i, cell, this, cellCoord, blueP);
		if(blueN->prop > 0)
			continue;
		
		bool fof = false;
		int nblueCuts = fCell.blueNodeFaceOnFront(i, cell, this, cellCoord, fof);
		
/// no blue-blue
		if(nblueCuts < 1)
			continue;
			
/// two blue-blue, blue, red-red all on front
		if(nblueCuts > 2 )
			continue;
				
		BccNode * redBlueN = fCell.addRedBlueEdgeNode(i, this, cellCoord);
		redBlueN->pos = (redP + blueP) * .5f;
		
		redBlueN->prop = 7;
		
	}
}

}
