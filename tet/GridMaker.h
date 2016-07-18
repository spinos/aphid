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
#include "tetrahedron_math.h"

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
	
	BccTetraGrid * grid();
	
protected:
	const ITetrahedron * tetra(const int & i) const;
	
	template<typename Tn>
	void extractGridNodesIn(Tn * dst, int & i,
						aphid::sdb::Array<int, BccNode> * cell) {
		cell->begin();
		while(!cell->end() ) {
			
			BccNode * n = cell->value();
			if(n->index > -1) {
				Tn * d = &dst[i++];
				d->pos = n->pos;
			}
			
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
	
	template<typename Tn>
	void obtainGridNodeValIn(const Tn * src,
							aphid::sdb::Array<int, BccNode> * cell) {
		cell->begin();
		while(!cell->end() ) {
			
			BccNode * n = cell->value();
			
			n->val = src[n->index].val;
			
			cell->next();
		}
	}
	
	template<typename Tn>
	void obtainGridNodeVal(const Tn * src) {
		m_grid.begin();
		while(!m_grid.end() ) {
			
			obtainGridNodeValIn(src, m_grid.value() );
			m_grid.next();
		}
	}
	
	template<typename Tn>
	bool checkTetraVolumeExt(const Tn * src) const
	{
		float mnvol = 1e20f, mxvol = -1e20f, vol;
		aphid::Vector3F p[4];
		const int n = numTetrahedrons();
		int i = 0;
		for(;i<n;++i) {
			const ITetrahedron * t = m_tets[i];
			if(!t) continue;
			
			p[0] = src[t->iv0].pos;
			p[1] = src[t->iv1].pos;
			p[2] = src[t->iv2].pos;
			p[3] = src[t->iv3].pos;
			
			vol = aphid::tetrahedronVolume(p);
			if(mnvol > vol)
				mnvol = vol;
			if(mxvol < vol)
				mxvol = vol;
				
		}

		std::cout<<"\n min/max tetrahedron volume: "<<mnvol<<" / "<<mxvol;
		if(mnvol < 0.f) {
			std::cout<<"\n [ERROR] negative volume";
			return false;
		}
			
		return true;
	}
	
private:
	void internalClear();
	
};

}
#endif