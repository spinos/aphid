/*
 *  MassiveTetraGridTriangulation.h
 *  
 *	triangulate each cell
 *
 *  Created by jian zhang on 2/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_MASSIVE_TETRA_GRID_TRIANGULATION_H
#define APH_TTG_MASSIVE_TETRA_GRID_TRIANGULATION_H

#include <ttg/TetraGridTriangulation.h>
#include <ttg/TetrahedronGrid.h>
#include <sdb/GridClosestToPoint.h>
#include <sdb/LodGrid.h>

namespace aphid {

namespace ttg {

template <typename Tv, typename Tg>
class MassiveTetraGridTriangulation : public TetraGridTriangulation<Tv, Tg> {

typedef TetraGridTriangulation<Tv, Tg> ParentTyp;

	ATriangleMesh * m_frontMesh;
	
public:
	MassiveTetraGridTriangulation();
	virtual ~MassiveTetraGridTriangulation();
	
/// field and mesh each cell on front
	template<typename Tintersect, typename Tclosest>
	void triangulate(Tintersect & fintersect, 
					Tclosest & fclosest, 
					CalcDistanceProfile & profile);
	
	const ATriangleMesh * frontMesh() const;
	
protected:

private:
/// any sign changed on six edges
	bool isTetrahedronOnFront(const float & a,
						const float & b,
						const float & c,
						const float & d) const;
	
};

template <typename Tv, typename Tg>
MassiveTetraGridTriangulation<Tv, Tg>::MassiveTetraGridTriangulation()
{
	TetrahedronGridUtil<5 > tu4;
	m_frontMesh = 0;
}

template <typename Tv, typename Tg>
MassiveTetraGridTriangulation<Tv, Tg>::~MassiveTetraGridTriangulation()
{}

template <typename Tv, typename Tg> 
template<typename Tintersect, typename Tclosest>
void MassiveTetraGridTriangulation<Tv, Tg>::triangulate(Tintersect & fintersect, 
						Tclosest & fclosest,
						CalcDistanceProfile & profile)
{
	profile.offset = fintersect.getBBox().getLongestDistance() * .000977f;
	sdb::LodGrid lodG;
	
typedef sdb::GridClosestToPoint<sdb::LodGrid, sdb::LodCell, sdb::LodNode > SelGridTyp;

	SelGridTyp selGrid(&lodG);
	selGrid.setMaxSelectLevel(4);
	
	sdb::AdaptiveGridDivideProfle subdprof;
	subdprof.setLevels(0, 4);
	subdprof.setToDivideAllChild(false);
	
	const Tg * g = ParentTyp::grid();
	const DistanceNode * nds = ParentTyp::field()->nodes();
    
	const int & nt = g->numCells();
	cvx::Tetrahedron atet;
	TetraGridTriangulation<Tv, TetrahedronGrid<Tv, 5 > >  amesher;
	
	for(int i=0;i<nt;++i) {
		const sdb::Coord4 & itet = g->cellVertices(i);
        if(isTetrahedronOnFront(nds[itet.x].val, 
                             nds[itet.y].val, 
							nds[itet.z].val, 
                            nds[itet.w].val) ) {
							
			std::cout<<"\n MassiveTetraGridTriangulation::triangulate "<<i;
			g->getCell(atet, i);
			
			TetrahedronGrid<Tv, 5 > * tetg = new TetrahedronGrid<Tv, 5 >(atet, 0);
			amesher.setGrid(tetg);
			
			BoundingBox tbx = atet.calculateBBox();
			tbx.expand(.1f);
			
			lodG.fillBox(tbx, tbx.getLongestDistance() );
			lodG. template subdivideToLevel<Tintersect>(fintersect, subdprof);
			lodG. template insertNodeAtLevel<Tclosest, 4 >(4, fclosest);
	
			amesher.field()-> template calculateDistance<SelGridTyp>(tetg, &selGrid, profile);
			amesher.triangulate();
			
			m_frontMesh = new ATriangleMesh;
			amesher.dumpFrontTriangleMesh(m_frontMesh);
			m_frontMesh->calculateVertexNormals();
			
			std::cout<<"\n n tri "<<m_frontMesh->numTriangles();
	
			delete tetg;
			
			if(m_frontMesh->numTriangles() ) break;
		}
	}
}

template <typename Tv, typename Tg>
const ATriangleMesh * MassiveTetraGridTriangulation<Tv, Tg>::frontMesh() const
{ return m_frontMesh; }

template <typename Tv, typename Tg>
bool MassiveTetraGridTriangulation<Tv, Tg>::isTetrahedronOnFront(const float & a,
						const float & b,
						const float & c,
						const float & d) const
{
	if(a * b < 0.f) {
		return true;
	}
	if(a * c < 0.f) {
		return true;
	}
	if(a * d < 0.f) {
		return true;
	}
	if(b * c < 0.f) {
		return true;
	}
	if(c * d < 0.f) {
		return true;
	}
	if(d * b < 0.f) {
		return true;
	}
	return false;
}

}

}
#endif