/*
 *  BccTetraGrid.cpp
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <BccTetraGrid.h>
#include "Convexity.h"
#include <line_math.h>
#include "tetrahedron_math.h"

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
	const float r = gridSize() * .25f;
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

void BccTetraGrid::moveBlueToFront(const aphid::Vector3F & cellCenter,
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
		BccNode * xi = fCell.blueNode6(i+6, cell, this, cellCoord);
/// already on front
		if(xi->prop > 0) 
			continue;
			
		fCell.getCellCorner(blueP, i, gz);
		samples->getClosest(closestP, d, blueP);
		
		if(fCell.moveBlueTo(closestP, blueP, 1.7f * r) ) {
			
		// std::cout<<"\n move blue";
		//xi->pos = closestP;
			xi->prop = 3;
			xi->pos = closestP;
		}
		
	}
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
		
/// already cut ?		
		BccNode * node = fCell.redRedNode(i, cell, this, cellCoord);

		if(!node) {
/// cut any way
			node = fCell.addFaceNode(i, this, cellCoord);
		}
		
/// already on front
		if(node->prop > 0)
			continue;
			
			fCell.facePosition(q, facePs, nfaceP, blueOnFront, i, cell, this, cellCoord);
			
			BccNode * nend = fCell.faceNode(i, cell, this, cellCoord);
			redOnFront = (redn->prop > -1 && nend->prop > -1);
			antiRedP = nend->pos;
			
/// choose longer one
		if(redP.distanceTo(antiRedP) * .5f > redP.distanceTo(q) )
			q = (redP + antiRedP) * .5f;

/// cut at mid
		node->pos = q;
				
/// move to sample if possible
			samples->getClosest(closestP, d, q);
			
			if(fCell.checkSplitFace(closestP, redP, antiRedP, r, i/2, facePs, nfaceP) ) {

/// limit distance moved		
				if(closestP.distanceTo(node->pos) < 1.5f * r) {
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
	const float r = gridSize() * .3f;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	
	Vector3F p1, p2, closestP;
	float d;

	int i=0, prop1, prop2;
	for(;i<12;++i) {

/// already cut ?
		BccNode * node = fCell.blueBlueNode(i, cell, this, cellCoord);

		if(node ) 
			continue;
			
/// add anyway
		node = fCell.addEdgeNode(i, this, cellCoord);
		
		BccNode * b1 = fCell.blueBlueEdgeNode(i, 0, cell, this, cellCoord);
		p1 = b1->pos;
		prop1 = b1->prop;
		
		BccNode * b2 = fCell.blueBlueEdgeNode(i, 1, cell, this, cellCoord);
		p2 = b2->pos;
		prop2 = b2->prop;

/// blue mid		
		node->pos = (p1 + p2) * .5f;
			
/// straddle blue on front
		if(prop1 > 0 && prop2 > 0) {
			node->prop = BccCell::NCyan;
		}

/// free cyan		
		if(prop1 < 0 && prop2 < 0) {
			samples->getClosest(closestP, d, node->pos);
		
			if(closestP.distanceTo(node->pos) < r)
				node->prop = BccCell::NCyan;
		}
	}
}

void BccTetraGrid::cutRedBlueEdges(const Vector3F & cellCenter,
					const sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{return;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	
	BccNode * redN = fCell.redNode(this, cellCoord);
		
	const Vector3F redP = redN->pos;
	
	const float r = gridSize() * .25f;
	Vector3F closestP;
	float d;
	
	
	Vector3F blueP, cutP0, cutP1, faceP;
	int nedgeCut, nfaceCut;
/// for each face 
	int i=0, j;
	for(;i<6;++i) {
		
		BccNode * fn = fCell.redRedNode(i, cell, this, cellCoord);
		//std::cout<<"\n facen"<<fn->key;
			if(fn->key>14999 && fn->prop < 0) {
			if(fCell.faceClosed(i, cell, this, cellCoord, faceP) ) {
			std::cout<<"\n face closed";
			fn->pos = faceP;
			fn->prop = 5;
			}
		}
		
/// for each blue
		for(j=0; j<4;++j) {
			BccNode * blueN = fCell.faceVaryBlueNode(i, j, cell, this, cellCoord);
/// blue on front
			if(blueN->prop > 0)
				continue;

/// already cut
			BccNode * rbn = fCell.faceVaryRedBlueNode(i, j, cell, this, cellCoord);
			if(rbn)
				continue;
				
			if(fCell.straddleBlueCut(i, j, cell, this, cellCoord,
							cutP0, cutP1) < 2 )
				continue;
				
			blueP = blueN->pos;
			BccNode * redBlueN = fCell.addFaceVaryRedBlueEdgeNode(i, j, this, cellCoord);
			redBlueN->pos = (cutP0 + cutP1) * .5f;
			redBlueN->prop = 7;
			
		
		}
		
	}
}

void BccTetraGrid::moveRedToFront(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
		
/// limit min red distance to faces
	const float r = gridSize() * .22f;
	BccNode * redN = cell->find(15);
	float d;
	Vector3F closestP;
	samples->getClosest(closestP, d, redN->pos);
		
	if(fCell.checkMoveRed(closestP, r, cell, this, cellCoord) ) {
		redN->pos = closestP;
		redN->prop = BccCell::NRed;
	}
}

void BccTetraGrid::cutAndWrap(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{
/// split distance
	const float r = gridSize() * .125f;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	BccNode * redN = cell->find(15);
	
	Vector3F tetv[4];
	BccNode nodeJ[3];
	tetv[0] = redN->pos;
	
/// per tetra
	int i = 0, j;
	for(i;i<48;++i) {
		
		fCell.getTetraYellowBlueCyan(i, cell, this, cellCoord, nodeJ[0], nodeJ[1], nodeJ[2]);

		tetv[1] = nodeJ[0].pos;
		tetv[2] = nodeJ[1].pos;
		tetv[3] = nodeJ[2].pos;
		
		RefineOption op = getRefineOpt(*redN, nodeJ[0], nodeJ[1], nodeJ[2]);
			
		if(op == RoSplitRedYellow) {
			processSplitRedYellow(i, redN, &nodeJ[0],
								fCell, cell, cellCoord, samples, r);
		}
		
		else if(op == RoSplitRedBlueOrCyan) { 
			
		}
	}
}

bool BccTetraGrid::tetraIsOpenOnFront(const BccNode & a,
					const BccNode & b,
					const BccNode & c,
					const BccNode & d) const
{	
	if(b.prop > 0 && c.prop > 0 && d.prop > 0)
		return false;
		
	int n = 0;
	if(a.prop > 0)
		n++;
		
	if(b.prop > 0)
		n++;
		
	if(c.prop > 0)
		n++;
		
	if(d.prop > 0)
		n++;
		
	return (n>1 && n<4);
}

bool BccTetraGrid::tetraEncloseSample(aphid::Vector3F & sampleP, 
					const aphid::Vector3F * v,
					const ClosestSampleTest * samples) const
{
	Vector3F q;
	float fd;
	q = (v[0] + v[1] + v[2] + v[3]) * .25f;
	samples->getClosest(sampleP, fd, q);
	
	return Convexity::CheckInsideTetra(v, sampleP);
}

bool BccTetraGrid::edgeIntersectFront(const BccNode & a,
					const BccNode & b,
					const ClosestSampleTest * samples,
					Vector3F & q) const
{
/// on front
	if(a.prop > 0 && b.prop >0) 
		return false;
		
	float d;
	Vector3F pa, pb, va, vb, mid;
	if(a.prop > 0) {
		pa = a.pos;
		va.setZero();
	}
	else {
		samples->getClosest(pa, d, a.pos);
		va = pa - a.pos;
		va.normalize();
	}
	
	if(b.prop > 0) {
		pb = b.pos;
		vb.setZero();
	}
	else {
		samples->getClosest(pb, d, b.pos);
		vb = pb - b.pos;
		vb.normalize();
	}
		
/// both end on the same side
	if(va.dot(vb) > 0.f)
		return false;

/// close to average		
	samples->getClosest(mid, d, (pa + pb) * .5f);
	
	if(!distancePointLineSegment(d, mid, a.pos, b.pos) )
		return false;
	
/// p on seg
	projectPointLineSegment(q, d, mid, a.pos, b.pos);
	
	return true;
}

bool BccTetraGrid::vertexCloseToFront(const BccNode & a,
					const aphid::Vector3F * v,
					const ClosestSampleTest * samples,
					const float & r,
					aphid::Vector3F & q) const
{
	float d;
	samples->getClosest(q, d, a.pos);
	//if(!Convexity::CheckInsideTetra(v, q) )
	//	return false;
		
	if(q.distanceTo(a.pos) > r)
		return false;
		
	return true;
}

BccTetraGrid::RefineOption BccTetraGrid::getRefineOpt(const BccNode & a,
					const BccNode & b,
					const BccNode & c,
					const BccNode & d) const
{
	if(a.prop > 0 && b.prop > 0 && c.prop > 0 && d.prop > 0)
		return RoNone;

/// blue and cyan on front	
	if(a.prop < 0 && b.prop < 0 && c.prop > 0 && d.prop > 0)
		return RoSplitRedYellow;
		
/// red-yellow on front, move cyan when edge has no blue on front
//	if(a.prop > 0 && b.prop > 0 && c.prop < 0 && d.prop < 0)
//		return RoMoveCyan;

/// yellow-blue or yellow-cyan on front, split red and the other		
	if(b.prop > 0 && (c.prop * d.prop < 0) )
		return RoSplitRedBlueOrCyan;
		
	return RoNone;
}

void BccTetraGrid::processSplitRedYellow(const int & i,
					BccNode * redN,
					BccNode * yellowN,
					const BccCell & fCell,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples,
					const float & r)
{
	Vector3F q;
	
	if(edgeIntersectFront(*redN, *yellowN, samples, q) )
		fCell.cutTetraRedBlueCyanYellow(i, 0, cell, this, cellCoord, q, r);
	
}

}
