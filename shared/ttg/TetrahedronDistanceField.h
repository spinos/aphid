/*
 *  TetrahedronDistanceField.h
 *  
 *  T as grid type
 *  Created by zhang on 17-1-30.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_TETRAHEDRON_DISTANCE_FIELD_H
#define APH_TTG_TETRAHEDRON_DISTANCE_FIELD_H

#include <graph/BaseDistanceField.h>
#include <ttg/TetraGridEdgeMap.h>
#include <ttg/TetraDistance.h>
#include <geom/ConvexShape.h>
#include <vector>
#include <map>

namespace aphid {

namespace ttg {

template<typename T>
class TetrahedronDistanceField : public BaseDistanceField {

public:
    TetrahedronDistanceField();
    virtual ~TetrahedronDistanceField();
    
    void buildGraph(T * grid, TetraGridEdgeMap<T > * edgeMap);
    void updateGrid(T * grid) const;
    bool getCutEdgePos(Vector3F & pdst,
                const int & v1, const int & v2) const;
    
    template<typename Tf>
    void calculateDistance(T * grid, Tf * intersectF,
                            const Vector3F & refPoint,
                            const Vector3F & refDirection,
                            const float & offset) 
    {
		std::cout<<"\n TetrahedronDistanceField::calculateDistance";
		std::cout.flush();
		
        resetNodes(1e20f, sdf::StBackGround, sdf::StUnknown);
        uncutEdges();
        
/// for each cell
        cvx::Tetrahedron atet;
        Vector3F tcen;
        float trad;
        const int & nt = grid->numCells();
        for(int i=0;i<nt;++i) {
            grid->getCell(atet, i);
            
            atet.getCenterRadius(tcen, trad);
            if(!intersectF->select(tcen, trad) ) {
                continue;
            }
			
/// approximate as node distance to plane
            TetraDistance cutDist(atet);
            cutDist.compute(intersectF, trad * 2.f, offset);
            
            const sdb::Coord4 & tetv = grid->cellVertices(i);
            const float * dist = cutDist.result();
            const bool * validD = cutDist.isValid();
            if(validD[0]) {
                setNodeDistance(tetv.x, dist[0]);
            }
            if(validD[1]) {
                setNodeDistance(tetv.y, dist[1]);
            }
            if(validD[2]) {
                setNodeDistance(tetv.z, dist[2]);
            }
            if(validD[3]) {
                setNodeDistance(tetv.w, dist[3]);
            }
            
            if(validD[0] && validD[1]) {
                cutEdge(tetv.x, tetv.y, dist[0], dist[1]);
            }
            
            if(validD[1] && validD[2]) {
                cutEdge(tetv.y, tetv.z, dist[1], dist[2]);
            }
            
            if(validD[2] && validD[0]) {
                cutEdge(tetv.z, tetv.x, dist[2], dist[0]);
            }
            
            if(validD[0] && validD[3]) {
                cutEdge(tetv.x, tetv.w, dist[0], dist[3]);
            }
            
            if(validD[1] && validD[3]) {
                cutEdge(tetv.y, tetv.w, dist[1], dist[3]);
            }
            
            if(validD[2] && validD[3]) {
                cutEdge(tetv.z, tetv.w, dist[2], dist[3]);
            }
        }
        
/// propagate distance to all nodes        
        fastMarchingMethod();
        
        int iFar = nodeFarthestFrom(refPoint, refDirection);
/// visit out nodes
        marchOutside(iFar);
/// unvisited nodes are inside
        setFarNodeInside();
/// merge short edges
        snapToFront();
		std::cout<<"\n done.";
    }
    
protected:

private:
    void pushIndices(const std::vector<int> & a,
							std::vector<int> & b);
    void extractGridPos(T * grid);
    
};

template<typename T>
TetrahedronDistanceField<T>::TetrahedronDistanceField()
{}

template<typename T>
TetrahedronDistanceField<T>::~TetrahedronDistanceField()
{}

template<typename T>
void TetrahedronDistanceField<T>::buildGraph(T * grid, 
                            TetraGridEdgeMap<T > * edgeMap)
{
    std::map<int, std::vector<int> > vvemap;
	
	int c = 0;
	edgeMap->begin();
	while(!edgeMap->end() ) {
	
		int v0 = edgeMap->key().x;
		vvemap[v0].push_back(c);
		
		int v1 = edgeMap->key().y;
		vvemap[v1].push_back(c);
		
		c++;
		edgeMap->next();
	}
	
	std::vector<int> edgeBegins;
	std::vector<int> edgeInds;
	
	int nvve = 0;
	std::map<int, std::vector<int> >::iterator it = vvemap.begin();
	for(;it!=vvemap.end();++it) {
		edgeBegins.push_back(nvve);
		
		pushIndices(it->second, edgeInds);
		nvve += (it->second).size();
		
		it->second.clear();
	}
    
    int nv = grid->numPoints();
	int ne = edgeMap->size();
	int ni = edgeInds.size();
	BaseDistanceField::create(nv, ne, ni);
    
    extractGridPos(grid);
	extractEdges(edgeMap);
	extractEdgeBegins(edgeBegins);
	extractEdgeIndices(edgeInds);
    
    vvemap.clear();
	edgeBegins.clear();
	edgeInds.clear();
	
    calculateEdgeLength();
    
}

template<typename T>
void TetrahedronDistanceField<T>::pushIndices(const std::vector<int> & a,
							std::vector<int> & b)
{
	std::vector<int>::const_iterator it = a.begin();
	for(;it!=a.end();++it) {
		b.push_back(*it);
    }
}

template<typename T>
void TetrahedronDistanceField<T>::extractGridPos(T * grid)
{   
    DistanceNode * dst = nodes();
    const int n = grid->numPoints();
    for(int i=0;i<n;++i) {
        dst[i].pos = grid->pos(i);
    }
}

template<typename T>
void TetrahedronDistanceField<T>::updateGrid(T * grid) const
{
    const int & nv = grid->numPoints();
    for(int i=0;i<nv;++i) {
        grid->setNodeDistance(nodes()[i].val, i);
    }
}

template<typename T>
bool TetrahedronDistanceField<T>::getCutEdgePos(Vector3F & pdst,
                                const int & v1, const int & v2) const
{
    int ei = edgeIndex(v1, v2);
    const IDistanceEdge & e = edges()[ei];
    if(e.cx < 0.f) {
        return false;
    }
    if(e.vi.x == v1) {
        pdst = nodes()[v2].pos * e.cx + nodes()[v1].pos * (1.f - e.cx);
    } else {
        pdst = nodes()[v1].pos * e.cx + nodes()[v2].pos * (1.f - e.cx);
    }
    return true;
}

}

}
#endif
