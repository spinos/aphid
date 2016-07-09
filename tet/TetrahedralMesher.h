/*
 *  TetrahedralMesher.h
 *  foo
 *
 *  Created by jian zhang on 6/26/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_TETRAHEDRAL_MESHER_H
#define TTG_TETRAHEDRAL_MESHER_H

#include "BccTetraGrid.h"

struct Float4;

namespace ttg {

class TetrahedralMesher {

	BccTetraGrid m_grid;
	aphid::Vector3F * m_X;
	int * m_prop;
	int m_N;
	std::vector<ITetrahedron *> m_tets;
	aphid::sdb::Array<aphid::sdb::Coord3, IFace > m_frontFaces;
	aphid::sdb::Array<aphid::sdb::Coord3, ClosestSampleTest> m_frontCellCoords;
	
public:
	TetrahedralMesher();
	virtual ~TetrahedralMesher();
	
	void clear();
	void setH(const float & x);
	void addCell(const aphid::Vector3F & p);
	void addFrontCell(const aphid::Vector3F & p,
				const std::vector<aphid::Vector3F> & samples);
	void buildGrid();
	int finishGrid();
	int numNodes();
	void setN(const int & x);
	void extractGridPosProp();
	int buildMesh();
	bool addPoint(const int & vi, bool & topologyChanged);
	ITetrahedron * searchTet(const aphid::Vector3F & p, Float4 * coord);
	bool checkConnectivity();
	void checkTetraVolume();
	
	const int & N() const;
	aphid::Vector3F * X();
	const aphid::Vector3F * X() const;
	const int * prop() const;
	const ITetrahedron * tetrahedron(const int & vi) const;
	const ITetrahedron * frontTetrahedron(const int & vi,
									int nfront = 1,
									int nmaxfront = 4) const;
	int numTetrahedrons();
	int buildFrontFaces();
	aphid::sdb::Array<aphid::sdb::Coord3, IFace > * frontFaces();
	
protected:

private:
	int countFrontVetices(const ITetrahedron * t) const;
	void addFrontFaces(const ITetrahedron * t);
	void processCells();
	void cutFaces();
	void cutEdges();
	void cutRedBlue();
	void moveBlue();
	void closeRed();
	
};

}

#endif
