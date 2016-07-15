/*
 *  GridMaker.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_GRID_MAKER_H
#define TTG_GRID_MAKER_H

#include "BccTetraGrid.h"

namespace ttg {

class GridMaker {

	BccTetraGrid m_grid;
	std::vector<ITetrahedron *> m_tets;
	
public:
	GridMaker();
	virtual ~GridMaker();
	
	void setH(const float & x);
	void addCell(const aphid::Vector3F & p);
	void buildGrid();
	void buildMesh();
	int numTetrahedrons() const;
	
protected:
	BccTetraGrid * grid();
	const ITetrahedron * tetra(const int & i) const;
	
	template<typename Tn>
	void extractGridNodesIn(Tn * dst, int & i,
						aphid::sdb::Array<int, BccNode> * cell) {
		cell->begin();
		while(!cell->end() ) {
			
			BccNode * n = cell->value();
			
			Tn * d = &dst[i++];
			d->pos = n->pos;
			
			cell->next();
		}
	}
	
	template<typename Tn>
	void extractGridNodes(Tn * dst) {
		int i = 0;
		m_grid.begin();
		while(!m_grid.end() ) {
			
			extractGridNodesIn(dst, i, m_grid.value() );
			m_grid.next();
		}
	}
	
private:
	void internalClear();
	
};

}
#endif