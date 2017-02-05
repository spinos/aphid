/*
 *  TetraGridTriangulation.h
 *  
 *  Tv is node value type
 *  Tg is grid type
 *  Created by zhang on 17-1-30.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_TETRA_GRID_TRIANGULATION_H
#define APH_TTG_TETRA_GRID_TRIANGULATION_H

#include <ttg/TetraGridEdgeMap.h>
#include <ttg/TetrahedronDistanceField.h>
#include <ttg/RedBlueRefine.h>
#include <geom/ATriangleMesh.h>

namespace aphid {

template <typename Tv, typename Tg>
class TetraGridTriangulation {
    
public: 
typedef ttg::TetrahedronDistanceField<Tg > FieldT;

private:
	Tg * m_tg;
    FieldT * m_field;

    TetraGridEdgeMap<Tg > * m_edgeMap;
    Vector3F * m_cutEdgePos;
    sdb::Sequence<sdb::Coord3 > m_frontTriangleMap;
    
public:    
    TetraGridTriangulation();
    virtual ~TetraGridTriangulation();
    
    void setGrid(Tg * g);
    
    void triangulate();
    
    TetraGridEdgeMap<Tg > & gridEdges();
    int numFrontTriangles();
    void dumpFrontTriangleMesh(ATriangleMesh * trimesh);
    
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
                    
    void getEdgeValuePos(float & va, float & vb,
                    Vector3F & pa, Vector3F & pb,
                    const int & v1, const int & v2);
                    
    struct VertexInd {
        int key;
        int ind;
    };
    
    void addAVertex(sdb::Array<int, VertexInd > & vertmap,
                    const int & k);
                    
};

template <typename Tv, typename Tg>
TetraGridTriangulation<Tv, Tg>::TetraGridTriangulation()
{ 
    m_tg = 0;
    m_edgeMap = 0;
    m_cutEdgePos = 0;
}

template <typename Tv, typename Tg>
TetraGridTriangulation<Tv, Tg>::~TetraGridTriangulation()
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

template <typename Tv, typename Tg>
void TetraGridTriangulation<Tv, Tg>::setGrid(Tg * g)
{
    m_tg = g;
    m_edgeMap = new TetraGridEdgeMap<Tg >(m_tg);
    m_field = new FieldT;
    m_field->buildGraph(g, m_edgeMap );
    
}

template <typename Tv, typename Tg>
void TetraGridTriangulation<Tv, Tg>::triangulate()
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

template <typename Tv, typename Tg>
TetraGridEdgeMap<Tg > & TetraGridTriangulation<Tv, Tg>::gridEdges()
{ return *m_edgeMap; }

template <typename Tv, typename Tg>
int TetraGridTriangulation<Tv, Tg>::numFrontTriangles()
{ return m_frontTriangleMap.size(); }

template <typename Tv, typename Tg>
int TetraGridTriangulation<Tv, Tg>::cutEdgeInd(int & numCuts, const int & v0, const int & v1)
{
    int  * e = m_edgeMap->findEdge(v0, v1);
    if(* e < 0) {
        * e = numCuts;
        numCuts++;
    }
    return *e;
}

template <typename Tv, typename Tg>
void TetraGridTriangulation<Tv, Tg>::getEdgeValuePos(float & va, float & vb,
                    Vector3F & pa, Vector3F & pb,
                    const int & v1, const int & v2)
{
    const DistanceNode & n1 = m_field->nodes()[v1];
    va = n1.val;
    pa = n1.pos;
    const DistanceNode & n2 = m_field->nodes()[v2];
    vb = n2.val;
    pb = n2.pos;
}

template <typename Tv, typename Tg>
void TetraGridTriangulation<Tv, Tg>::cutEdges(int & numCuts,
                    ttg::RedBlueRefine & refiner, 
                    const sdb::Coord4 & itet)
{
    if(!refiner.hasOption() ) {
		return;
    }
	
    float da, db;
    Vector3F pa, pb;
	int icut;
	if(refiner.needSplitRedEdge(0) ) {
        icut = cutEdgeInd(numCuts, itet.x, itet.y);        
        if(!m_field->getCutEdgePos(m_cutEdgePos[icut], itet.x, itet.y) ) {
            getEdgeValuePos(da, db, pa, pb, itet.x, itet.y);
            m_cutEdgePos[icut] = refiner.splitPos(da, db, pa, pb);
        }
		refiner.splitRedEdge(0,icut | MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitRedEdge(1) ) {
		icut = cutEdgeInd(numCuts, itet.z, itet.w);		
        if(!m_field->getCutEdgePos(m_cutEdgePos[icut], itet.z, itet.w) ) {
            getEdgeValuePos(da, db, pa, pb, itet.z, itet.w);
            m_cutEdgePos[icut] = refiner.splitPos(da, db, pa, pb);
        }
        refiner.splitRedEdge(1,icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(0) ) {
		icut = cutEdgeInd(numCuts, itet.x, itet.z);        
        if(!m_field->getCutEdgePos(m_cutEdgePos[icut], itet.x, itet.z) ) {
            getEdgeValuePos(da, db, pa, pb, itet.x, itet.z);
            m_cutEdgePos[icut] = refiner.splitPos(da, db, pa, pb);
        }
		refiner.splitBlueEdge(0, icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(1) ) {
		icut = cutEdgeInd(numCuts, itet.x, itet.w);        
        if(!m_field->getCutEdgePos(m_cutEdgePos[icut], itet.x, itet.w) ) {
            getEdgeValuePos(da, db, pa, pb, itet.x, itet.w);
            m_cutEdgePos[icut] = refiner.splitPos(da, db, pa, pb);
        }
		refiner.splitBlueEdge(1, icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(2) ) {
		icut = cutEdgeInd(numCuts, itet.y, itet.z);        
        if(!m_field->getCutEdgePos(m_cutEdgePos[icut], itet.y, itet.z) ) {
            getEdgeValuePos(da, db, pa, pb, itet.y, itet.z);
            m_cutEdgePos[icut] = refiner.splitPos(da, db, pa, pb);
        }
		refiner.splitBlueEdge(2, icut| MEncode, m_cutEdgePos[icut]);
	}
	
	if(refiner.needSplitBlueEdge(3) ) {
		icut = cutEdgeInd(numCuts, itet.y, itet.w);        
        if(!m_field->getCutEdgePos(m_cutEdgePos[icut], itet.y, itet.w) ) {
            getEdgeValuePos(da, db, pa, pb, itet.y, itet.w);
            m_cutEdgePos[icut] = refiner.splitPos(da, db, pa, pb);
        }
		refiner.splitBlueEdge(3, icut| MEncode, m_cutEdgePos[icut]);
	}
}

template <typename Tv, typename Tg>
ttg::TetrahedronDistanceField<Tg > * TetraGridTriangulation<Tv, Tg>::field()
{ return m_field; }

template <typename Tv, typename Tg>
const ttg::TetrahedronDistanceField<Tg > * TetraGridTriangulation<Tv, Tg>::field() const
{ return m_field; }

template <typename Tv, typename Tg>
void TetraGridTriangulation<Tv, Tg>::addAVertex(sdb::Array<int, VertexInd > & vertmap,
                    const int & k)
{
    VertexInd * v = new VertexInd;
    v->key = k;
    v->ind = -1;
    vertmap.insert(k, v);
}

template <typename Tv, typename Tg>
void TetraGridTriangulation<Tv, Tg>::dumpFrontTriangleMesh(ATriangleMesh * trimesh)
{
    const unsigned nt = numFrontTriangles();
    
    sdb::Array<int, VertexInd > vertmap;
    m_frontTriangleMap.begin();
	while(!m_frontTriangleMap.end() ) {
        const sdb::Coord3 & k = m_frontTriangleMap.key();
        
        if(!vertmap.find(k.x) ) {
            addAVertex(vertmap, k.x);
        }
        
        if(!vertmap.find(k.y) ) {
            addAVertex(vertmap, k.y);
        }
        
        if(!vertmap.find(k.z) ) {
            addAVertex(vertmap, k.z);
        }

		m_frontTriangleMap.next();
	}
    
    const unsigned np = vertmap.size();
    
    int acc = 0;
    vertmap.begin();
    while(!vertmap.end() ) {
        VertexInd * vi = vertmap.value();
        if(vi->ind < 0) {
            vi->ind = acc;
            acc++;
        }
        
		vertmap.next();
	}
    
    trimesh->create(np, nt);
    
    unsigned * indDst = trimesh->indices();
    Vector3F * pntDst = trimesh->points();
    
    const DistanceNode * nds = m_field->nodes(); 
    int itri = 0;
    m_frontTriangleMap.begin();
	while(!m_frontTriangleMap.end() ) {
        const sdb::Coord3 & k = m_frontTriangleMap.key();
        
        const VertexInd * v1 = vertmap.find(k.x);
        const VertexInd * v2 = vertmap.find(k.y);
        const VertexInd * v3 = vertmap.find(k.z);
        
        indDst[itri * 3] = v1->ind;
        indDst[itri * 3 + 1] = v2->ind;
        indDst[itri * 3 + 2] = v3->ind;
        itri++;
        
        if(k.x < MDecode) {
            pntDst[v1->ind] = nds[k.x].pos;
        } else {
            pntDst[v1->ind] = m_cutEdgePos[k.x & MDecode];
        }
        if(k.y < MDecode) {
            pntDst[v2->ind] = nds[k.y].pos;
        } else {
            pntDst[v2->ind] = m_cutEdgePos[k.y & MDecode];
        }
        if(k.z < MDecode) {
            pntDst[v3->ind] = nds[k.z].pos;
        } else {
            pntDst[v3->ind] = m_cutEdgePos[k.z & MDecode];
        }

		m_frontTriangleMap.next();
	}
    
    vertmap.clear();
}

}
#endif

