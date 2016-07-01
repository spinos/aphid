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
		c.addNodes(this, gridCoord((const float *)c.centerP() ) );
	}
	cells.clear();
	countNodes();
	
}

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

void BccTetraGrid::extractNodePositions(Vector3F * dest)
{
	begin();
	while(!end() ) {
		
		extractNodePositionsIn(dest, value() );
		next();
	}
}

void BccTetraGrid::extractNodePositionsIn(Vector3F * dest,
									sdb::Array<int, BccNode> * cell)
{
	cell->begin();
	while(!cell->end() ) {
		dest[cell->value()->index] = cell->value()->pos;
		
		cell->next();
	}
}

void BccTetraGrid::buildTetrahedrons(std::vector<ITetrahedron *> & dest)
{
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
					const float & d,
					const Vector3F & pos,
					Vector3F * X,
					int * prop)
{	
	BccCell cell(cellCenter);
	int xi = cell.indexToNode15(this, gridCoord((const float *)&cellCenter ) );
			
/// limit in which red can move
	if(d > gridSize() * .25f) {
		return false;
	}
		
	X[xi] = pos;
	prop[xi] = 4;
	
	return true;
}

void BccTetraGrid::moveBlueNodes(const Vector3F & cellCenter,
					const aphid::Vector3F & redP,
					const std::vector<Vector3F> & samples,
					aphid::Vector3F * X,
					int * prop)
{
	const float gz = gridSize();
	const float r = gz * .25f;
	BccCell cell(cellCenter);
	const sdb::Coord3 cellCoord = gridCoord((const float *)&cellCenter );
	int xi;
	float d;
	Vector3F blueP, closestP;
/// for each blue
	int i=0;
	for(;i<8;++i) {
		xi = cell.indexToBlueNode(i, this, cellCoord, gz, blueP);
/// already moved
		if(prop[xi] > 0) 
			continue;
			
		GetClosest<Vector3F>(closestP, d, blueP, samples);
		
		if(isBlueCloseToRed(closestP, blueP, redP, r) ) {
			continue;
		}
			
		//std::cout<<"\n move blue"<<xi;
		X[xi] = closestP;
		prop[xi] = 3;
	}
}

/// limit minimum angles
bool BccTetraGrid::isBlueCloseToRed(const aphid::Vector3F & p,
					const aphid::Vector3F & blueP,
					const aphid::Vector3F & redP,
					const float & r) const
{
/// limit in octan
	Vector3F dp = p - blueP;
	if(Absolute<float>(dp.x) > 2.f * r) return true;
	if(Absolute<float>(dp.y) > 2.f * r) return true;
	if(Absolute<float>(dp.z) > 2.f * r) return true;
/// limit to red
	dp = p - redP;
	if(Absolute<float>(dp.x) < r) return true;
	if(Absolute<float>(dp.y) < r) return true;
	if(Absolute<float>(dp.z) < r) return true;
	return false;
}

void BccTetraGrid::moveRedNodeIn(const aphid::Vector3F & cellCenter,
					const Vector3F & pos,
					aphid::Vector3F * X,
					int * prop)
{
	BccCell cell(cellCenter);
	int xi;
	if(cell.moveNode15(xi, this, gridCoord((const float *)&cellCenter ),
					pos) ) {
		X[xi] = pos;
		prop[xi] = 1;
	}
}

void BccTetraGrid::smoothBlueNodeIn(const aphid::Vector3F & cellCenter,
					aphid::Vector3F * X)
{
	BccCell cell(cellCenter);
	int xi[9];
	Vector3F pos;
/// for each vertex node
	int i = 6;
	for(;i<14;++i) {
		if(cell.getVertexNodeIndices(i, xi, this, gridCoord((const float *)&cellCenter ) ) ) {
/// average of eight nodes
			X[xi[0]] = (X[xi[1]]
						+ X[xi[2]]
						+ X[xi[3]]
						+ X[xi[4]]
						+ X[xi[5]]
						+ X[xi[6]]
						+ X[xi[7]]
						+ X[xi[8]]) * .125f;
		}
	}
}

}
