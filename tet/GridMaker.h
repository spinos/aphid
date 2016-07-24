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
	std::vector<aphid::sdb::Coord3 > m_triInds;
	
public:
	GridMaker();
	virtual ~GridMaker();
	
	void setH(const float & x);
	void addCell(const aphid::Vector3F & p);
	void buildGrid();
	void buildMesh();
	int numTetrahedrons() const;
	int numTriangles() const;
	
	BccTetraGrid * grid();
	
protected:
	const ITetrahedron * tetra(const int & i) const;
	const aphid::sdb::Coord3 & triangleInd(const int & i) const;
	
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
	
};

}
#endif