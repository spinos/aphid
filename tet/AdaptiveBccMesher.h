/*
 *  AdaptiveBccMesher.h
 *  
 *	contains grid tetrahedrons triangles
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_ADAPTIVE_BCC_MESHER_H
#define TTG_ADAPTIVE_BCC_MESHER_H

#include "AdaptiveBccGrid3.h"
#include "tetrahedron_math.h"

namespace ttg {

class AdaptiveBccMesher {

	AdaptiveBccGrid3 m_grid;
	std::vector<ITetrahedron *> m_tets;
	int m_numVert;
	
public:
	AdaptiveBccMesher();
	virtual ~AdaptiveBccMesher();
	
/// reset grid w level0 cell size and fill the box at level 0
	void fillBox(const aphid::BoundingBox & b,
				const float & h);
	
	void buildGrid();
	void buildMesh();
	int numTetrahedrons() const;
	const int & numVertices() const;
	
	AdaptiveBccGrid3 * grid();
	const std::vector<ITetrahedron *> & tetrahedrons() const;
	
protected:
	const ITetrahedron * tetra(const int & i) const;
	
/// for each cell divied, must have same level neighbor cell on six faces
/// level change cross face < 2
	void enforceBoundary(std::vector<aphid::sdb::Coord4 > & ks);

private:
	void internalClear();
	void clearTetra();
	
};

}
#endif