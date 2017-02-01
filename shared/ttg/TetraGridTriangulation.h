/*
 *  TetraGridTriangulation.h
 *  
 *  Tv is node value type
 *  N is order of grid
 *  Created by zhang on 17-1-30.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_TETRA_GRID_TRIANGULATION_H
#define APH_TTG_TETRA_GRID_TRIANGULATION_H

#include <ttg/TetrahedronGrid.h>
#include <ttg/TetraGridEdgeMap.h>
#include <ttg/TetrahedronDistanceField.h>
#include <ttg/RedBlueRefine.h>

namespace aphid {

template <typename Tv, int N>
class TetraGridTriangulation {
    
public: 
typedef TetrahedronGrid<Tv, N> GridT;
typedef ttg::TetrahedronDistanceField<GridT > FieldT;

private:
	GridT * m_tg;
    FieldT * m_field;

    TetraGridEdgeMap<GridT > * m_edgeMap;
    Vector3F * m_cutEdgePos;
    sdb::Sequence<sdb::Coord3 > m_frontTriangleMap;
    
public:    
    TetraGridTriangulation();
    virtual ~TetraGridTriangulation();
    
    void setGrid(GridT * g);
    
    void triangulate();
    
    TetraGridEdgeMap<TetrahedronGrid<Tv, N> > & gridEdges();
    int numFrontTriangles();
    void extractFrontTriangles(Vector3F * vs);
    
    FieldT * field();
    const FieldT * field() const;
    
protected:

private:
     enum CutIndMask {
		MEncode = 1048576,
		MDecode = 1048575,
	};
    
    int cutEdgeInd(int & numCuts, 
                    const int & v0, 
                    const int & v1);

    void cutEdges(int & numCuts,
                    ttg::RedBlueRefine & refiner, 
                    const sdb::Coord4 & itet);
};

template <typename Tv, int N>
TetraGridTriangulation<Tv, N>::TetraGridTriangulation()
{ 
    m_tg = 0;
    m_edgeMap = 0;
    m_cutEdgePos = 0;
}

template <typename Tv, int N>
TetraGridTriangulation<Tv, N>::~TetraGridTriangulation()
{
    if(m_tg) {
        delete m_tg;
    }
    
    if(m_edgeMap) {
        m_edgeMap->clear();
        delete m_edgeMap;
    }
    
    if(m_cutEdgePos) {
        delete[] m_cutEdgePos;
    }
}

template <typename Tv, int N>
void TetraGridTriangulation<Tv, N>::setGrid(GridT * g)
{
    m_tg = g;
    m_edgeMap = new TetraGridEdgeMap<GridT >(m_tg);
    m_field = new FieldT;
    m_field->buildGraph(g, m_edgeMap );
    
}

template <typename Tv, int N>
void TetraGridTriangulation<Tv, N>::triangulate()
{
    const int ne = m_edgeMap->size();
    m_cutEdgePos = new Vector3F[ne];
    
    const DistanceNode * nds = m_field->nodes();
    
    m_frontTriangleMap.clear();
    int numCuts = 0;
    ttg::RedBlueRefine rbr;
    const int nt = m_tg->numCells();
    for(int i=0;i<nt;++i) {
        const sdb::Coord4 & itet = m_tg->cellVertices(i);
        rbr.set(itet.x, itet.y, itet.z, itet.w);
		rbr.evaluateDistance(nds[itet.x].val, 
                             nds[itet.y].val, 
							nds[itet.z].val, 
                            nds[itet.w].val );
		rbr.estimateNormal(nds[itet.x].pos,
                            nds[itet.y].pos,
                            nds[itet.z].pos,
                            nds[itet.w].pos );
        cutEdges(numCuts, rbr, itet);
        rbr.refine();
        
        const int nft = rbr.numFrontTriangles();
		for(int j=0; j<nft; ++j) {
			const ttg::IFace * fj = rbr.frontTriangle(j);
			if(!m_frontTriangleMap.find(fj->key) ) {
				m_frontTriangleMap.insert(fj->key);
			}
		}
    }
    
}

template <typename Tv, int N>
TetraGridEdgeMap<TetrahedronGrid<Tv, N> > & TetraGridTriangulation<Tv, N>::gridEdges()
{ return *m_edgeMap; }

template <typename Tv, int N>
int TetraGridTriangulation<Tv, N>::numFrontTriangles()
{ return m_frontTriangleMap.size(); }

template <typename Tv, int N>
int TetraGridTriangulation<Tv, N>::cutEdgeInd(int & numCuts, const int & v0, const int & v1)
{
    int  * e = m_edgeMap->findEdge(v0, v1);
    if(* e < 0) {
        * e = numCuts;
        numCuts++;
    }
    return *e;
}

template <typename Tv, int N>
void TetraGridTriangulation<Tv, N>::cutEdges(int & numCuts,
                    ttg::RedBlueRefine & refiner, 
                    const sdb::Coord4 & itet)
{
    if(!refiner.hasOption() )
		return;
	
    float d0, d1;
    Vector3F p0, p1;
	int icut;
	if(refiner.needSplitRedEdge(0) ) {
        icut = cutEdgeInd(numCuts, itet.x, itet.y);        
        m_field->getCutEdgePos(m_cutEdgePos[icut], itet.x, itet.y);
		refiner.splitRedEdge(0,icut | MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitRedEdge(1) ) {
		icut = cutEdgeInd(numCuts, itet.z, itet.w);		
        m_field->getCutEdgePos(m_cutEdgePos[icut], itet.z, itet.w);
        refiner.splitRedEdge(1,icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(0) ) {
		icut = cutEdgeInd(numCuts, itet.x, itet.z);        
        m_field->getCutEdgePos(m_cutEdgePos[icut], itet.x, itet.z);
		refiner.splitBlueEdge(0, icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(1) ) {
		icut = cutEdgeInd(numCuts, itet.x, itet.w);        
        m_field->getCutEdgePos(m_cutEdgePos[icut], itet.x, itet.w);
		refiner.splitBlueEdge(1, icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(2) ) {
		icut = cutEdgeInd(numCuts, itet.y, itet.z);        
        m_field->getCutEdgePos(m_cutEdgePos[icut], itet.y, itet.z);
		refiner.splitBlueEdge(2, icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(3) ) {
		icut = cutEdgeInd(numCuts, itet.y, itet.w);        
        m_field->getCutEdgePos(m_cutEdgePos[icut], itet.y, itet.w);
		refiner.splitBlueEdge(3, icut| MEncode, m_cutEdgePos[icut]);
	}
}

template <typename Tv, int N>
void TetraGridTriangulation<Tv, N>::extractFrontTriangles(Vector3F * vs)
{
    const DistanceNode * nds = m_field->nodes(); 
    int itri = 0;
    m_frontTriangleMap.begin();
	while(!m_frontTriangleMap.end() ) {
        const sdb::Coord3 & k = m_frontTriangleMap.key();
        if(k.x < MDecode) {
            vs[itri * 3] = nds[k.x].pos;
        } else {
            vs[itri * 3] = m_cutEdgePos[k.x & MDecode];
        }
        if(k.y < MDecode) {
            vs[itri * 3 + 1] = nds[k.y].pos;
        } else {
            vs[itri * 3 + 1] = m_cutEdgePos[k.y & MDecode];
        }
        if(k.z < MDecode) {
            vs[itri * 3 + 2] = nds[k.z].pos;
        } else {
            vs[itri * 3 + 2] = m_cutEdgePos[k.z & MDecode];
        }
		itri++; 

		m_frontTriangleMap.next();
	}
}

template <typename Tv, int N>
ttg::TetrahedronDistanceField<TetrahedronGrid<Tv, N> > * TetraGridTriangulation<Tv, N>::field()
{ return m_field; }

template <typename Tv, int N>
const ttg::TetrahedronDistanceField<TetrahedronGrid<Tv, N> > * TetraGridTriangulation<Tv, N>::field() const
{ return m_field; }

}
#endif

