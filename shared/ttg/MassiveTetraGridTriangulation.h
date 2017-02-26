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

typedef TetraGridTriangulation<Tv, Tg> MesherTyp;
typedef GenericHexagonGrid<Tv> CoarseGridTyp;
typedef HexagonDistanceField<CoarseGridTyp> CoarseFieldTyp;
typedef TetrahedronDistanceField<Tg > FineFieldTyp;

	CoarseGridTyp m_coarseGrid;
	CoarseFieldTyp m_coarseField;
	
	std::vector<FineFieldTyp *> m_fineFields;
/// result of triangulation
	std::vector<ATriangleMesh *> m_frontMeshes;
	
public:
	struct Profile {
		BoundingBox coarsGridBox;
		float coarseCellSize;
		int coarseGridSubdivLevel;
		int fineGridSubdivLevel;
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
	
	int numFineFields() const;
	const FineFieldTyp * fineField(int i) const;
	
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
	for(int i=0;i<m_fineFields.size();++i) {
		delete m_fineFields[i];
	}
	m_fineFields.clear();
	
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
	subdprof.setToDivideAllChild(true);
	
	bccg. template subdivideToLevel<Tintersect>(fintersect, subdprof);
	bccg.build();
	
	bccg. template buildHexagonGrid <CoarseGridTyp> (&m_coarseGrid, prof.coarseGridSubdivLevel);
	
	m_coarseField.buildGraph(&m_coarseGrid);
	
	CalcDistanceProfile distprof;
	distprof.referencePoint = prof.coarsGridBox.lowCorner();
	distprof.direction.set(1.f, 1.f, 1.f);
	distprof.offset = 0.f;
    distprof.snapDistance = 0.f;
	
	m_coarseField. template calculateDistance<Tclosest>(&m_coarseGrid, &fclosest, distprof);

	const float gz0 = bccg.levelCellSize(prof.coarseGridSubdivLevel);
	subdprof.setLevels(0, prof.fineGridSubdivLevel);
		
	BoundingBox cellBx;
	cvx::Hexagon ahexa;
	const int & nc = m_coarseGrid.numCells();
	for(int i=0;i<nc;++i ) {
		m_coarseGrid.getCell(ahexa, i);
		cellBx = ahexa.calculateBBox();
		
		if(!fclosest.select(cellBx) ) {
			continue;
		}
		
		AdaptiveBccGrid3 abccg;
		abccg.resetBox(cellBx, gz0);
		abccg. template subdivideToLevel<Tintersect>(fintersect, subdprof);
		abccg.build();
		
		TetraMeshBuilder teter;
		Tg tetg;
			
		teter.buildMesh(&tetg, &abccg);
		
		FineFieldTyp * afld = new FineFieldTyp;
	
		MesherTyp mesher;
		mesher.setGridField(&tetg, afld);
		
		CalcDistanceProfile adistprof;
		adistprof.referencePoint = cellBx.center();
		adistprof.direction = fclosest.aggregatedNormal();
		adistprof.offset = .1f;
		adistprof.snapDistance = .1f;
			
		afld-> template calculateDistance<Tclosest>(&tetg, &fclosest, adistprof);
		
		m_fineFields.push_back(afld);
		
		mesher.triangulate();
		
		if(mesher.numFrontTriangles() > 0) {
					
			ATriangleMesh * amesh = new ATriangleMesh;
			mesher.dumpFrontTriangleMesh(amesh);
			amesh->calculateVertexNormals();
	
			m_frontMeshes.push_back(amesh);
		}
		
		if(m_frontMeshes.size() > 2) break;
		
	}

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

template <typename Tv, typename Tg>
int MassiveTetraGridTriangulation<Tv, Tg>::numFineFields() const
{ return m_fineFields.size(); }

template <typename Tv, typename Tg>
const TetrahedronDistanceField<Tg > * MassiveTetraGridTriangulation<Tv, Tg>::fineField(int i) const
{ return m_fineFields[i]; }

}

}
#endif