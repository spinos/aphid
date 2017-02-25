/*
 *  MassiveTetraGridTriangulation.h
 *  
 *	subdivide, triangulate each cell
 *
 *  Created by jian zhang on 2/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_MASSIVE_TETRA_GRID_TRIANGULATION_H
#define APH_TTG_MASSIVE_TETRA_GRID_TRIANGULATION_H

#include <ttg/TetraGridTriangulation.h>
#include <ttg/GenericHexagonGrid.h>
#include <ttg/AdaptiveBccGrid3.h>
#include <ttg/HexagonDistanceField.h>
#include <sdb/GridClosestToPoint.h>
#include <sdb/LodGrid.h>

namespace aphid {

namespace ttg {

template <typename Tv, typename Tg>
class MassiveTetraGridTriangulation {

typedef TetraGridTriangulation<Tv, Tg> ParentTyp;
typedef GenericHexagonGrid<Tv> CoarseGridTyp;
typedef HexagonDistanceField<CoarseGridTyp> CoarseFieldTyp;

	CoarseGridTyp m_coarseGrid;
	CoarseFieldTyp m_coarseField;
	
/// result of triangulation
	std::vector<ATriangleMesh *> m_frontMeshes;
	
public:
	struct Profile {
		BoundingBox coarsGridBox;
		float coarseCellSize;
		int coarseGridSubdivLevel;
		
	};

	MassiveTetraGridTriangulation();
	virtual ~MassiveTetraGridTriangulation();
	
/// field and mesh each cell on front
	template<typename Tintersect, typename Tclosest>
	void triangulate(Tintersect & fintersect, 
					Tclosest & fclosest, 
					Profile & prof);
	
	int numFrontMeshes() const;
	const ATriangleMesh * frontMesh(int i) const;
	
	const GenericHexagonGrid<Tv> * coarseGrid() const;
	const CoarseFieldTyp * coarseField() const;
	
protected:

private:
	void internalClear();
	
};

template <typename Tv, typename Tg>
MassiveTetraGridTriangulation<Tv, Tg>::MassiveTetraGridTriangulation()
{}

template <typename Tv, typename Tg>
MassiveTetraGridTriangulation<Tv, Tg>::~MassiveTetraGridTriangulation()
{ 
	internalClear();
}

template <typename Tv, typename Tg>
void MassiveTetraGridTriangulation<Tv, Tg>::internalClear()
{
	for(int i=0;i<m_frontMeshes.size();++i) {
		delete m_frontMeshes[i];
	}
	m_frontMeshes.clear();
	
}

template <typename Tv, typename Tg> 
template<typename Tintersect, typename Tclosest>
void MassiveTetraGridTriangulation<Tv, Tg>::triangulate(Tintersect & fintersect, 
						Tclosest & fclosest,
						Profile & prof)
{
	internalClear();
	
	AdaptiveBccGrid3 bccg;
	bccg.fillBox(prof.coarsGridBox, prof.coarseCellSize);
	
	sdb::AdaptiveGridDivideProfle subdprof;
	subdprof.setLevels(0, prof.coarseGridSubdivLevel);
	subdprof.setToDivideAllChild(false);
	
	bccg.subdivideToLevel<Tintersect>(fintersect, subdprof);
	bccg.build();
	
	bccg. template buildHexagonGrid <CoarseGridTyp> (&m_coarseGrid, prof.coarseGridSubdivLevel);
	
	m_coarseField.buildGraph(&m_coarseGrid);
	
	CalcDistanceProfile distprof;
	distprof.referencePoint = prof.coarsGridBox.lowCorner();
	distprof.direction.set(1.f, 1.f, 1.f);
	distprof.offset = 0.f;
    distprof.snapDistance = 0.5f;
	
	m_coarseField. template calculateDistance<Tclosest>(&m_coarseGrid, &fclosest, distprof);

	/*
	
	const float gz0 = m_coarseGrid.levelCellSize(profile.coarseGridSubdivLevel);
	subdprof.setLevels(0, profile.fineGridSubdivLevel);
	subdprof.setToDivideAllChild(true);
		
	BoundingBox cellBx;
	m_coarseGrid.begin();
	while(!m_coarseGrid.end() ) {
		
		if(m_coarseGrid.key() == profile.coarseGridSubdivLevel) {
			m_coarseGrid.getCellBBox(cellBx, m_coarseGrid.key() );
			
			ttg::AdaptiveBccGrid3 bccg;
			bccg.resetBox(cellBx, gz0);
			bccg.subdivideToLevel<FIntersectTyp>(ineng, subdprof);
			bccg.build();
			
			ttg::TetraMeshBuilder teter;
			ttg::GenericTetraGrid<Tv > tetg;
			
			teter.buildMesh(&tetg, &bccg);
	
			ttg::TetraGridTriangulation<Tv, ttg::GenericTetraGrid<Tv > > mesher;
			mesher.setGrid(&tetg);
			
			ttg::TetrahedronDistanceField<ttg::GenericTetraGrid<Tv > > * fld = mesher.field();
			fld-> template calculateDistance<Tclosest>(&tetg, &fclosest, profile);
			
			mesher.triangulate();
			
			if(mesher.numFrontTriangles() > 0) {
					
				ATriangleMesh * amesh = new ATriangleMesh;
				mesher.dumpFrontTriangleMesh(amesh);
				amesh->calculateVertexNormals();
		
				m_frontMeshes.push_back(amesh);
			}
			
		}
		m_coarseGrid.next();
	}
	
	profile.offset = .1f;
	
	const Tg * g = ParentTyp::grid();
	const DistanceNode * nds = ParentTyp::field()->nodes();
    
	const int & nt = g->numCells();
	cvx::Tetrahedron atet;
	TetraGridTriangulation<Tv, TetrahedronGrid<Tv, 4 > >  amesher;
	
	Vector3F tcen;
	float trad;
	
	for(int i=0;i<nt;++i) {
	
		const sdb::Coord4 & itet = g->cellVertices(i);
		if(isTetrahedronInside(nds[itet.x].val, 
                             nds[itet.y].val, 
							nds[itet.z].val, 
                            nds[itet.w].val) ) {
			continue;
		}
		
		g->getCell(atet, i);
		atet.getCenterRadius(tcen, trad);
		
		if(!fclosest.select(tcen, trad) ) {
			continue;
		}
		
		profile.referencePoint = tcen;
		const int vo = vertexOutside(itet, nds[itet.x].val, 
                             nds[itet.y].val, 
							nds[itet.z].val, 
                            nds[itet.w].val);
							
		profile.direction = nds[vo].pos - profile.referencePoint;
		profile.snapDistance = trad * .1f;
			
		TetrahedronGrid<Tv, 4 > * tetg = new TetrahedronGrid<Tv, 4 >(atet, 0);
		amesher.setGrid(tetg);
		
		amesher.field()-> template calculateDistance<Tclosest>(tetg, &fclosest, profile);
		amesher.triangulate();
		
		if(amesher.numFrontTriangles() < 1) {
			continue;
		}
		
		ATriangleMesh * amesh = new ATriangleMesh;
		amesher.dumpFrontTriangleMesh(amesh);
		amesh->calculateVertexNormals();
		
		m_frontMeshes.push_back(amesh);
		
		delete tetg;
			
		if(m_frontMeshes.size() > 99) {
			//break;
		}
			
	}
	*/
	std::cout<<"\n MassiveTetraGridTriangulation::triangulate n mesh "<<numFrontMeshes();
}

template <typename Tv, typename Tg>
const GenericHexagonGrid<Tv> * MassiveTetraGridTriangulation<Tv, Tg>::coarseGrid() const
{
	return &m_coarseGrid;
}

template <typename Tv, typename Tg>
const HexagonDistanceField<GenericHexagonGrid<Tv> > * MassiveTetraGridTriangulation<Tv, Tg>::coarseField() const
{
	return &m_coarseField;
}

template <typename Tv, typename Tg>
int MassiveTetraGridTriangulation<Tv, Tg>::numFrontMeshes() const
{ return m_frontMeshes.size(); }

template <typename Tv, typename Tg>
const ATriangleMesh * MassiveTetraGridTriangulation<Tv, Tg>::frontMesh(int i) const
{ return m_frontMeshes[i]; }

}

}
#endif