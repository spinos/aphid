/*
 *  BccCell.h
 *  foo
 *
 *  Created by jian zhang on 7/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "triangulation.h"
#include "tetrahedron_graph.h"
#include <WorldGrid.h>
#include <Array.h>

namespace ttg {

class ClosestSampleTest {

	aphid::Vector3F * m_smps;
	int m_N;
	
public:
	ClosestSampleTest(const std::vector<aphid::Vector3F> & src);
	virtual ~ClosestSampleTest();
	
	int getClosest(aphid::Vector3F & dst, float & d, 
				const aphid::Vector3F & toPnt) const;
				
	int getIntersect(aphid::Vector3F & dst, float & d, 
				const aphid::Vector3F & seg1,
				const aphid::Vector3F & seg2) const;
};

/// face shared by two tetra
template<typename T>
class STriangle {

public:
	STriangle() :
	ta(NULL),
	tb(NULL) 
	{}
	
	aphid::sdb::Coord3 key;
	T * ta;
	T * tb;
	
};

typedef aphid::sdb::Array<aphid::sdb::Coord3, STriangle<ITetrahedron> > STriangleArray;

class BccNode {
	
public:
	aphid::Vector3F pos;
	int prop;
	int key;
	int index;
};

class BccCell {

	aphid::Vector3F m_center;
	
	static float TwentySixNeighborOffset[26][3];
	static int SevenNeighborOnCorner[8][7];
	static int SixTetraFace[6][8];
	static int SixNeighborOnFace[6][4];
	
public:
	BccCell(const aphid::Vector3F &center );
	void addRedBlueNodes(aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const;
	void getNodePositions(aphid::Vector3F * dest,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const;
	void connectNodes(std::vector<ITetrahedron *> & dest,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					STriangleArray * faces) const;	
	bool moveNode(int & xi,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					const aphid::Vector3F & p,
					int * moved) const;
	int indexToNode15(aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const;
	BccNode * redNode(aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const;
	BccNode * blueNode(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					aphid::Vector3F & p) const;
	BccNode * redRedNode(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const;
	BccNode * faceNode(const int & i,
					aphid::sdb::Array<int, BccNode> * cell,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const;
	BccNode * addFaceNode(const int & i,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord);
	bool moveBlueTo(const aphid::Vector3F & p,
					const aphid::Vector3F & q,
					const float & r);
	bool moveFaceTo(const int & i,
					const aphid::Vector3F & p,
					const aphid::Vector3F & q,
					const aphid::Vector3F & redP,
					const float & r);
	int indexToBlueNode(const int & i,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					const float & cellSize,
					aphid::Vector3F & p) const;
	bool moveNode15(int & xi,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord,
					const aphid::Vector3F & p) const;
	bool getVertexNodeIndices(int vi, int * xi,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord);
	const aphid::Vector3F * centerP() const;
	const aphid::Vector3F facePosition(const int & i, const float & gz) const;
	
private:
	aphid::sdb::Coord3 neighborCoord(const aphid::sdb::Coord3 & cellCoord, int i) const;
	void neighborOffset(aphid::Vector3F * dest, int i) const;
	BccNode * findNeighborCorner(aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::Vector3F & pos, int icorner) const;
	int keyToCorner(const aphid::Vector3F & corner,
				const aphid::Vector3F & center) const;
	void getNodePosition(aphid::Vector3F * dest,
						const int & nodeI,
						const float & gridSize) const;
	void connectNodesOnFace(std::vector<ITetrahedron *> & dest,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					aphid::sdb::Array<int, BccNode> * cell,
					const aphid::sdb::Coord3 & cellCoord,
					int inode15, 
					int a,
					const int & iface,
					STriangleArray * faces) const;
	BccNode * findCornerNodeInNeighbor(const int & i,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const;
	BccNode * findRedRedNodeInNeighbor(const int & i,
					aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const;
	void addFace(STriangleArray * faces,
				int a, int b, int c,
				ITetrahedron * t) const;
	
};

}