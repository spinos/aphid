/*
 *  BccCell3.h
 *  foo
 *
 *  Created by jian zhang on 7/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Array.h>
#include <AdaptiveGrid3.h>
#include "BccCell.h"

namespace ttg {

class BccCell3 : public aphid::sdb::Array<int, BccNode > {

	typedef aphid::sdb::AdaptiveGrid3<BccCell3, BccNode, 5 > AdaptiveGridT;

	bool m_hasChild;
	BccCell3 * m_parentCell;
	int m_childI;
	
public:
	BccCell3(Entity * parent = NULL);
	virtual ~BccCell3();
	
	void setHasChild();
	void setParentCell(BccCell3 * x, const int & i);
	void insertRed(const aphid::Vector3F & pref);
	void insertBlue(const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	void insertYellow(const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	void insertCyan(const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode * findBlue(const aphid::Vector3F & pref);
	void insertFaceOnBoundary(const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	
	const bool & hasChild() const;
	
	BccNode * blueNode(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode * yellowNode(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode * cyanNode(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
					
	void connectTetrahedrons(std::vector<ITetrahedron *> & dest,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	
private:
	BccNode * findBlueNodeInNeighbor(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode * findCyanNodeInNeighbor(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode * derivedBlueNode(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
/// red, yellow, or face
	BccNode * faceNode(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	bool isFaceDivided(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	bool isEdgeDivided(const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	
/// c ---- d
/// \     /
///  \   /
///   \ /
///    b
	void addOneTetra(std::vector<ITetrahedron *> & dest,
					BccNode * A, BccNode * B, BccNode * C, BccNode * D);
/// c --x--- d
/// \   |   /
///  \  |  /
///   \ | /
///     b
/// i edge 0:11	
	void addTwoTetra(std::vector<ITetrahedron *> & dest,
					const int & i,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid,
					BccNode * A, BccNode * B, BccNode * C, BccNode * D);
/// c--- x---  d
/// \   /|\   /
///  \ / | \ /
///  f0  |  f1
///    \ | /
///      b
/// i face 0:6 j fv edge 0:3 k edge 0:11
	void addFourTetra(std::vector<ITetrahedron *> & dest,
					const int & i,
					const int & j,
					const int & k,
					const aphid::sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid,
					BccNode * A, BccNode * B, BccNode * C, BccNode * D);
															
};

}