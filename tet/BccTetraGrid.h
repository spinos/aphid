/*
 *  BccTetraGrid.h
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_BCC_TET_GRID_H
#define TTG_BCC_TET_GRID_H
#include "ClosestSampleTest.h"
#include "BccCell.h"

namespace ttg {

class BccTetraGrid : public aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > 
{
	
public:
	BccTetraGrid();
	virtual ~BccTetraGrid();
	
	void addBlueNodes(const aphid::Vector3F & cellCenter);
	void countNodes();
	void buildNodes();
	int numNodes();
	void extractNodePosProp(aphid::Vector3F * destPos,
					int * destProp);
	void buildTetrahedrons(std::vector<ITetrahedron *> & dest);
	void moveNodeIn(const aphid::Vector3F & cellCenter,
					const aphid::Vector3F * pos, 
					const int & n,
					aphid::Vector3F * X,
					int * prop);
	bool moveRedNodeTo(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const aphid::Vector3F & pos);
	void moveBlueToFront(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples);	
	void cutRedRedEdges(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples);	
	void cutBlueBlueEdges(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples);
	void cutRedBlueEdges(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples);
	void moveRedToFront(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples);
	void cutAndWrap(const aphid::Vector3F & cellCenter,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples);
	
protected:
	enum RefineOption {
		RoNone = 0,
		RoSplitRedYellow = 1,
		RoSplitRedBlueOrCyan = 2,
		RoSplitRedBlueAndCyan = 3,
		RoHalfYellowBlueOrCyan = 4
	};
	
private:
	void countNodesIn(aphid::sdb::Array<int, BccNode> * cell, int & c);
	
	void extractNodePosPropIn(aphid::Vector3F * destPos,
					int * destProp,
					aphid::sdb::Array<int, BccNode> * cell);
	bool tetraIsOpenOnFront(const BccNode & a,
					const BccNode & b,
					const BccNode & c,
					const BccNode & d) const;
	bool tetraEncloseSample(aphid::Vector3F & sampleP, 
					const aphid::Vector3F * v,
					const ClosestSampleTest * samples) const;
	bool edgeIntersectFront(const BccNode & a,
					const BccNode & b,
					const ClosestSampleTest * samples,
					aphid::Vector3F & q,
					const float & r) const;
	bool vertexCloseToFront(const BccNode & a,
					const aphid::Vector3F * v,
					const ClosestSampleTest * samples,
					const float & r,
					aphid::Vector3F & q) const;
	RefineOption getRefineOpt(const BccNode & redN,
					const BccNode & yellowN,
					BccNode ** bc,
					const BccCell & fCell) const;
	BccNode * processSplitRedYellow(const int & i,
					BccNode * redN,
					BccNode * yellowN,
					const BccCell & fCell,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples,
					const float & r);
	void wrapFace(const int & i,
					BccNode * redN,
					const BccCell & fCell,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples,
					const float & r);
	void wrapEdge(const int & i,
					BccNode * redN,
					const BccCell & fCell,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples,
					const float & r);
	void wrapVertex(const int & i,
					BccNode * redN,
					const BccCell & fCell,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples,
					const float & r);
	BccNode * processSplitRedBlueOrCyan(const int & i,
					BccNode * redN,
					BccNode ** bc,
					const BccCell & fCell,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples,
					const float & r);
	void processSplitRedBlueAndCyan(const int & i,
					BccNode * redN,
					BccNode ** bc,
					const BccCell & fCell,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples,
					const float & r);
	BccNode * splitFaceVaryEdge(const int & i,
					const int & j,
					BccNode * redN,
					BccNode * endN,
					const BccCell & fCell,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples,
					const float & r);
	void processHalfYellowBlueOrCyan(const int & i,
					BccNode * redN,
					BccNode * yellowN,
					BccNode ** bc,
					const BccCell & fCell,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					const ClosestSampleTest * samples,
					const float & r);
};

}

#endif