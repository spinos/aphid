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
			
			//if(blueOnFront<1 && !redOnFront) 
			//	toAdd = fCell.checkWedgeFace(redP, antiRedP, facePs, r);
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
	
	/*i=0;
	for(;i<6;++i) {
		fCell.checkFaceValume(i, cell, this, cellCoord);
	}
	
	return;
	if(redn->prop < 0) {
	samples->getClosest(closestP, d, redP);
		//std::cout<<"\n d/r"<<d/r;
		if(d< 1.5f * r) {
			std::cout<<"\n close red";
			redn->pos = closestP;
			redn->prop = 4;
		}
	}*/
}

void BccTetraGrid::cutBlueBlueEdges(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{
	const float gz = gridSize();
	const float r = gz * .25f;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	BccNode * redN = fCell.redNode(this, cellCoord);
	const Vector3F redP = redN->pos;
	
	int i, prop1, prop2, nredFront, nred, nyellowFront, nyellow;
	Vector3F p1, p2, q, closestP, yellowCenter;
	float d;

	i=0;
	for(;i<12;++i) {
/// already cut
		if(fCell.blueBlueNode(i, cell, this, cellCoord) )
			continue;
			
		BccNode * b1 = fCell.blueBlueEdgeNode(i, 0, cell, this, cellCoord);
		p1 = b1->pos;
		prop1 = b1->prop;
		
		BccNode * b2 = fCell.blueBlueEdgeNode(i, 1, cell, this, cellCoord);
		p2 = b2->pos;
		prop2 = b2->prop;
		
		//q = (p1 + p2) * .5f;
		
		fCell.edgeRedCenter(i, cell, this, cellCoord, q, nred, nredFront);
		if(nredFront < 4)
			q = (p1 + p2) * .5f;
			
		if(nred < 4)
			continue;
			 
		fCell.edgeYellowCenter(i, cell, this, cellCoord, yellowCenter, nyellow, nyellowFront);
		if(nyellowFront > 3)
			q = yellowCenter;
		
/// add anyway
		BccNode * node = fCell.addEdgeNode(i, this, cellCoord);
		node->pos = q;
/// both blue on front
		if(prop1 > 0 && prop2 > 0)
			node->prop = 6;
			
		if(nredFront > 3 || nyellowFront > 3)
			node->prop = 6;
		
		bool toMove = samples->getClosest(closestP, d, q) > -1;
		
		if(toMove) {
			toMove = fCell.checkSplitBlueBlueEdge(closestP, redP, p1, p2, r, i,
					cell, this, cellCoord);
		}
		
/// limit distance moved
		if(toMove) {
			//if(closestP.distanceTo(q) > r) {
			//	std::cout<<"\n d > r "<<closestP.distanceTo(q);
			//}
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

void BccTetraGrid::moveFaces(const Vector3F & cellCenter,
					const sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{

	const float r = gridSize() * .25f;
	BccCell fCell(cellCenter);
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	
	BccNode * redN = fCell.redNode(this, cellCoord);
	if(redN->prop > 0)
		return;
		
	const Vector3F redP = redN->pos;
	
	Vector3F closestP, blueP;
	float d;
/// for each face
	int i=0;
	for(;i<6;++i) {
		BccNode * rrn = fCell.redRedNode(i, cell, this, cellCoord);
		if(!rrn)
			continue;
			
		if(rrn->prop > 0)
			continue;
			
		samples->getClosest(closestP, d, rrn->pos);
		
		if(d< .5f * r) {
			rrn->pos = closestP;
			rrn->prop = 5;
		}
		
	}
	
	int nedgeCut, nfaceCut;
/// for each vertex
	i=0;
	for(;i<8;++i) {
		BccNode * blueN = fCell.blueNode(i, cell, this, cellCoord, blueP);
		if(blueN->prop > 0)
			continue;
			
		fCell.blueNodeConnectToFront(nedgeCut, nfaceCut, i, cell, this, cellCoord);
		
		if(nedgeCut < 3 && nfaceCut < 3)
			continue;
			
		samples->getClosest(closestP, d, blueN->pos);
		
		if(d< .5f * r) {
			blueN->pos = closestP;
			blueN->prop = 3;
		}
	}
	
	
/*
	BccNode * redN = fCell.redNode(this, cellCoord);
	if(redN->prop > 0)
		return;
		
	
	
	i=0;
	for(;i<8;++i) {
		BccNode * blueN = fCell.blueNode(i, cell, this, cellCoord, blueP);
		if(blueN->prop > 0)
			continue;
		
		fCell.blueNodeConnectToFront(nedgeCut, nfaceCut, i, cell, this, cellCoord);
		
		if(nedgeCut < 3)
			continue;
			
		
	}
*/
}

void BccTetraGrid::moveRedToYellowCenter(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples)
{
	BccCell fCell(cellCenter);
	BccNode * redN = fCell.redNode(this, cellCoord);
	if(redN->prop > 0)
		return;
		
	const float r = gridSize() * .2f;
	sdb::Array<int, BccNode> * cell = findCell(cellCoord );
	
	Vector3F q;
	int i = 0;
	for(;i<3;++i) {
		if(fCell.yellowFaceOnFront(i, cell, this, cellCoord, q) ) {
			if(q.distanceTo(redN->pos) <r ) {
			std::cout<<"\n close red to yellow face";
			redN->pos = q;
			redN->prop = 4;
			return;
			}
		}
	}
}

}
