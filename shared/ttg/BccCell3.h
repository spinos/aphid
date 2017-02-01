/*
 *  BccCell3.h
 *  ttg
 *
 *  Created by jian zhang on 7/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_BCC_CELL_3_H
#define APH_TTG_BCC_CELL_3_H

#include <sdb/Array.h>
#include <sdb/AdaptiveGrid3.h>
#include "RedBlueRefine.h"

namespace aphid {

namespace ttg {

class BccNode3 {
	
public:
	Vector3F pos;
	float val;
	int prop;
	int key;
	int index;
};

class BccCell3 : public sdb::Array<int, BccNode3 > {

	typedef sdb::AdaptiveGrid3<BccCell3, BccNode3, 10 > AdaptiveGridT;

	bool m_hasChild;
	BccCell3 * m_parentCell;
	int m_childI;
	
public:
	BccCell3(Entity * parent = NULL);
	virtual ~BccCell3();
	
	void setHasChild();
	void setParentCell(BccCell3 * x, const int & i);
	void insertRed(const Vector3F & pref);
	void insertBlue(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	void insertYellow(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	void insertCyan(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode3 * findBlue(const Vector3F & pref);
	void insertFaceOnBoundary(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	
	const bool & hasChild() const;
	
	BccNode3 * blueNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode3 * yellowNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode3 * cyanNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
					
	void connectTetrahedrons(std::vector<ITetrahedron *> & dest,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	
/// node value sign changed
	bool isFront(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
/// all node value negative
	bool isInterior(const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
					
private:
	void findRedValueFrontBlue(BccNode3 * redN,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode3 * findBlueNodeInNeighbor(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode3 * findCyanNodeInNeighbor(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	BccNode3 * derivedBlueNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
/// red, yellow, or face
	BccNode3 * faceNode(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	bool isFaceDivided(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	bool isEdgeDivided(const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid);
	
/// c ---- d
/// \     /
///  \   /
///   \ /
///    b
	void addOneTetra(std::vector<ITetrahedron *> & dest,
					BccNode3 * A, BccNode3 * B, BccNode3 * C, BccNode3 * D);
/// c --x--- d
/// \   |   /
///  \  |  /
///   \ | /
///     b
/// i edge 0:11	
	void addTwoTetra(std::vector<ITetrahedron *> & dest,
					const int & i,
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid,
					BccNode3 * A, BccNode3 * B, BccNode3 * C, BccNode3 * D);
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
					const sdb::Coord4 & cellCoord,
					AdaptiveGridT * grid,
					BccNode3 * A, BccNode3 * B, BccNode3 * C, BccNode3 * D);
	
};

}

}
#endif
