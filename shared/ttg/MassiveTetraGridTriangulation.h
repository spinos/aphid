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

/// result of triangulation
	std::vector<ATriangleMesh *> m_frontMeshes;
	
public:
	MassiveTetraGridTriangulation();
	virtual ~MassiveTetraGridTriangulation();
	
/// field and mesh each cell on front
	template<typename Tintersect, typename Tclosest>
	void triangulate(Tintersect & fintersect, 
					Tclosest & fclosest, 
					CalcDistanceProfile & profile);
	
	int numFrontMeshes() const;
	const ATriangleMesh * frontMesh(int i) const;
	
protected:

private:
/// any sign changed on six edges
	bool isTetrahedronOnFront(const float & a,
						const float & b,
						const float & c,
						const float & d) const;
/// first value >= 0.f
	int vertexOutside(const sdb::Coord4 & itet,
						const float & a,
						const float & b,
						const float & c,
						const float & d) const;
	
};

template <typename Tv, typename Tg>
MassiveTetraGridTriangulation<Tv, Tg>::MassiveTetraGridTriangulation()
{
	TetrahedronGridUtil<4 > tu4;
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
	float gz0 = fintersect.getBBox().getLongestDistance() / 32.f;
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
	TetraGridTriangulation<Tv, TetrahedronGrid<Tv, 4 > >  amesher;
	
	for(int i=0;i<nt;++i) {
		const sdb::Coord4 & itet = g->cellVertices(i);
        if(!isTetrahedronOnFront(nds[itet.x].val, 
                             nds[itet.y].val, 
							nds[itet.z].val, 
                            nds[itet.w].val) ) {
			continue;
		}
							
			std::cout<<"\n MassiveTetraGridTriangulation::triangulate "<<i;
			g->getCell(atet, i);
			
		profile.referencePoint = atet.getCenter();
		const int vo = vertexOutside(itet, nds[itet.x].val, 
                             nds[itet.y].val, 
							nds[itet.z].val, 
                            nds[itet.w].val);
							
		profile.direction = nds[vo].pos - profile.referencePoint;
			
			TetrahedronGrid<Tv, 4 > * tetg = new TetrahedronGrid<Tv, 4 >(atet, 0);
			amesher.setGrid(tetg);
			
			BoundingBox tbx = atet.calculateBBox();
			tbx.expand(1.f);
			
			lodG.fillBox(tbx,  gz0);
			lodG. template subdivideToLevel<Tintersect>(fintersect, subdprof);
			lodG. template insertNodeAtLevel<Tclosest, 4 >(4, fclosest);
	
			amesher.field()-> template calculateDistance<SelGridTyp>(tetg, &selGrid, profile);
			amesher.triangulate();
			
		if(amesher.numFrontTriangles() < 1) {
			continue;
		}
		
		ATriangleMesh * amesh = new ATriangleMesh;
		amesher.dumpFrontTriangleMesh(amesh);
		amesh->calculateVertexNormals();
		
		m_frontMeshes.push_back(amesh);
		
		std::cout<<"\n add n tri "<<amesh->numTriangles();

			delete tetg;
			
		if(m_frontMeshes.size() > 1) {
			break;
		}
			
	}
	
	std::cout<<"\n MassiveTetraGridTriangulation::triangulate n mesh "<<numFrontMeshes();
}

template <typename Tv, typename Tg>
int MassiveTetraGridTriangulation<Tv, Tg>::numFrontMeshes() const
{ return m_frontMeshes.size(); }

template <typename Tv, typename Tg>
const ATriangleMesh * MassiveTetraGridTriangulation<Tv, Tg>::frontMesh(int i) const
{ return m_frontMeshes[i]; }

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

template <typename Tv, typename Tg>
int MassiveTetraGridTriangulation<Tv, Tg>::vertexOutside(const sdb::Coord4 & itet,
						const float & a,
						const float & b,
						const float & c,
						const float & d) const
{
	if(a >= 0.f) {
		return itet.x;
	}
	if(b >= 0.f) {
		return itet.y;
	}
	if(c >= 0.f) {
		return itet.z;
	}
	return itet.w;
}

}

}
#endif