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
	std::vector<aphid::sdb::Coord3 > m_triInds;
	int m_numVert;
	
public:
	AdaptiveBccMesher();
	virtual ~AdaptiveBccMesher();
	
/// reset grid w level0 cell size
	void setH(const float & x);
	void addCell(const aphid::Vector3F & p);
	void buildGrid();
	void buildMesh();
	int numTetrahedrons() const;
	int numTriangles() const;
	const int & numVertices() const;
	
	AdaptiveBccGrid3 * grid();
	
	template<typename Tf>
	void subdivideGrid(Tf & distanceF, const int & level) {
		
/// track cells divided
		std::vector<aphid::sdb::Coord4 > divided;
		
		aphid::BoundingBox cellBox;
		
		m_grid.begin();
		while(!m_grid.end() ) {
			
			if(m_grid.key().w == level) {
				m_grid.getCellBBox(cellBox, m_grid.key() );
				//std::cout<<"\n cell box "<<cellBox;
				
				if(distanceF. template intersect<aphid::BoundingBox >(&cellBox) ) {
					
					m_grid.subdivideCell(m_grid.key() );
					
					divided.push_back(m_grid.key() );
				}
			}
			
			if(m_grid.key().w > level)
				break;
				
			m_grid.next();
		}
		
		if(level > 1)
			enforceBoundary(divided);
			
		divided.clear();
	}
	
protected:
	const ITetrahedron * tetra(const int & i) const;
	const aphid::sdb::Coord3 & triangleInd(const int & i) const;
	
	template<typename Tn>
	void extractGridNodesIn(Tn * dst,
						aphid::sdb::Array<int, BccNode> * cell) {
		cell->begin();
		while(!cell->end() ) {
			
			BccNode * n = cell->value();
			if(n->index > -1) {
				Tn * d = &dst[n->index];
				d->pos = n->pos;
			}
			
			cell->next();
		}
	}
	
	template<typename Tn>
	void extractGridNodes(Tn * dst) {
		m_grid.begin();
		while(!m_grid.end() ) {
			
			extractGridNodesIn(dst, m_grid.value() );
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
	
	void buildMesh1();
	
private:
	void internalClear();
	void clearTetra();
/// for each cell divied, must have same level neighbor cell on six faces
/// level change cross face < 2
	void enforceBoundary(const std::vector<aphid::sdb::Coord4 > & ks);
	
};

}
#endif