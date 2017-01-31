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
#include <ttg/RedBlueRefine.h>

namespace aphid {

template <typename Tv, int N>
class TetraGridTriangulation {
    
public: 
typedef TetrahedronGrid<Tv, N> GridT;

private:
	GridT * m_tg;

    TetraGridEdgeMap<GridT > * m_edgeMap;
    Vector3F * m_cutEdgePos;
    sdb::Sequence<sdb::Coord3 > m_frontTriangleMap;
    
public:    
    TetraGridTriangulation();
    virtual ~TetraGridTriangulation();
    
    void setGrid(TetrahedronGrid<Tv, N> * g);
    
    void triangulate();
    
    TetraGridEdgeMap<TetrahedronGrid<Tv, N> > & gridEdges();
    int numFrontTriangles();
    void extractFrontTriangles(Vector3F * vs);
    
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
void TetraGridTriangulation<Tv, N>::setGrid(TetrahedronGrid<Tv, N> * g)
{
    m_tg = g;
    m_edgeMap = new TetraGridEdgeMap<GridT >(m_tg);
}

template <typename Tv, int N>
void TetraGridTriangulation<Tv, N>::triangulate()
{
    const int ne = m_edgeMap->size();
    m_cutEdgePos = new Vector3F[ne];
    
    m_frontTriangleMap.clear();
    int numCuts = 0;
    ttg::RedBlueRefine rbr;
    const int nt = m_tg->numCells();
    for(int i=0;i<nt;++i) {
        const sdb::Coord4 & itet = m_tg->cellVertices(i);
        rbr.set(itet.x, itet.y, itet.z, itet.w);
		rbr.evaluateDistance(m_tg->value(itet.x)._distance, 
                             m_tg->value(itet.y)._distance, 
							m_tg->value(itet.z)._distance, 
                            m_tg->value(itet.w)._distance );
		rbr.estimateNormal(m_tg->pos(itet.x),
                            m_tg->pos(itet.y),
                            m_tg->pos(itet.z),
                            m_tg->pos(itet.w) );
        cutEdges(numCuts, rbr, itet);
        rbr.refine();
        
        const int nft = rbr.numFrontTriangles();
		for(int j=0; j<nft; ++j) {
			const ttg::IFace * fj = rbr.frontTriangle(j);
			//std::cout<<"\n ftri["<<j<<"] "<<fj->key;
            if(!m_frontTriangleMap.find(fj->key) ) {
				m_frontTriangleMap.insert(fj->key);
			}
		}
    }
    std::cout<<"\n n cut "<<numCuts;
    
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
        
        d0 = m_tg->value(itet.x)._distance;
        d1 = m_tg->value(itet.y)._distance;
        p0 = m_tg->pos(itet.x);
        p1 = m_tg->pos(itet.y);
        m_cutEdgePos[icut] = refiner.splitPos(d0, d1, p0, p1);
        
		refiner.splitRedEdge(0,icut | MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitRedEdge(1) ) {
		icut = cutEdgeInd(numCuts, itet.z, itet.w);
		
        d0 = m_tg->value(itet.z)._distance;
        d1 = m_tg->value(itet.w)._distance;
        p0 = m_tg->pos(itet.z);
        p1 = m_tg->pos(itet.w);
        m_cutEdgePos[icut] = refiner.splitPos(d0, d1, p0, p1);
        
        refiner.splitRedEdge(1,icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(0) ) {
		icut = cutEdgeInd(numCuts, itet.x, itet.z);
        
        d0 = m_tg->value(itet.x)._distance;
        d1 = m_tg->value(itet.z)._distance;
        p0 = m_tg->pos(itet.x);
        p1 = m_tg->pos(itet.z);
        m_cutEdgePos[icut] = refiner.splitPos(d0, d1, p0, p1);
        
		refiner.splitBlueEdge(0, icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(1) ) {
		icut = cutEdgeInd(numCuts, itet.x, itet.w);
        
        d0 = m_tg->value(itet.x)._distance;
        d1 = m_tg->value(itet.w)._distance;
        p0 = m_tg->pos(itet.x);
        p1 = m_tg->pos(itet.w);
        m_cutEdgePos[icut] = refiner.splitPos(d0, d1, p0, p1);
        
		refiner.splitBlueEdge(1, icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(2) ) {
		icut = cutEdgeInd(numCuts, itet.y, itet.z);
        
        d0 = m_tg->value(itet.y)._distance;
        d1 = m_tg->value(itet.z)._distance;
        p0 = m_tg->pos(itet.y);
        p1 = m_tg->pos(itet.z);
        m_cutEdgePos[icut] = refiner.splitPos(d0, d1, p0, p1);
        
		refiner.splitBlueEdge(2, icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(3) ) {
		icut = cutEdgeInd(numCuts, itet.y, itet.w);
        
        d0 = m_tg->value(itet.y)._distance;
        d1 = m_tg->value(itet.w)._distance;
        p0 = m_tg->pos(itet.y);
        p1 = m_tg->pos(itet.w);
        m_cutEdgePos[icut] = refiner.splitPos(d0, d1, p0, p1);
        
		refiner.splitBlueEdge(3, icut| MEncode, m_cutEdgePos[icut]);
	}
}

template <typename Tv, int N>
void TetraGridTriangulation<Tv, N>::extractFrontTriangles(Vector3F * vs)
{
    int itri = 0;
    m_frontTriangleMap.begin();
	while(!m_frontTriangleMap.end() ) {
        const sdb::Coord3 & k = m_frontTriangleMap.key();
        if(k.x < MDecode) {
            vs[itri * 3] = m_tg->pos(k.x);
        } else {
            vs[itri * 3] = m_cutEdgePos[k.x & MDecode];
        }
        if(k.y < MDecode) {
            vs[itri * 3 + 1] = m_tg->pos(k.y);
        } else {
            vs[itri * 3 + 1] = m_cutEdgePos[k.y & MDecode];
        }
        if(k.z < MDecode) {
            vs[itri * 3 + 2] = m_tg->pos(k.z);
        } else {
            vs[itri * 3 + 2] = m_cutEdgePos[k.z & MDecode];
        }
		itri++; 

		m_frontTriangleMap.next();
	}
}

}
#endif

