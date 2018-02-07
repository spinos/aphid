/*
 *  HexagonDistanceField.h
 *  
 *  T as grid type
 *  Created by zhang on 17-1-30.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_HEXAGON_DISTANCE_FIELD_H
#define APH_TTG_HEXAGON_DISTANCE_FIELD_H

#include <graph/BaseDistanceField.h>
#include <ttg/HexagonDistance.h>
#include <vector>
#include <map>

namespace aphid {

namespace ttg {

template<typename T>
class HexagonDistanceField : public BaseDistanceField {

public:
    HexagonDistanceField();
    virtual ~HexagonDistanceField();
    
    void buildGraph(const T * grid);
    
    template<typename Tf>
    void calculateDistance(T * grid, Tf * intersectF,
							CalcDistanceProfile & profile) 
    {
#if 1
		std::cout<<"\n HexagonDistanceField::calculateDistance";
		std::cout.flush();
#endif
        resetNodes(1e20f, sdf::StBackGround, sdf::StUnknown);
        uncutEdges();
        
        HexagonDistance ahexa;
        Vector3F tcen;
        float trad;
        const int & nt = grid->numCells();
        for(int i=0;i<nt;++i) {
            grid->getCell(ahexa, i);            
            ahexa.getCenterRadius(tcen, trad);
			
			if(!intersectF->select(tcen, trad) ) {
				continue;
			}
			
            ahexa.compute(intersectF, trad, profile.offset);
            
            const int * hexav = grid->cellVertices(i);
            const float * dist = ahexa.result();
            const bool * validD = ahexa.isValid();
			
			for(int j=0;j<8;++j) {
				if(validD[j]) {
					setNodeDistance(hexav[j], dist[j]);
				}
			}
			
        }
        
/// propagate distance to all nodes        
        fastMarchingMethod();
        
        int iFar = nodeFarthestFrom(profile.referencePoint, profile.direction);
/// visit out nodes
        marchOutside(iFar);
/// unvisited nodes are inside
        setFarNodeInside();
/// merge short edges
        snapToFrontByDistance(profile.snapDistance);
#if 0
		std::cout<<"\n done.";
#endif
    }
	
/// based on distance on corners
	Vector3F getCellNormal(const T * grid, const int & idx,
						const Vector3F & pcenter) const;
    
protected:

private:
    
    void extractGridPos(const T * grid);
    
};

template<typename T>
HexagonDistanceField<T>::HexagonDistanceField()
{}

template<typename T>
HexagonDistanceField<T>::~HexagonDistanceField()
{}

template<typename T>
void HexagonDistanceField<T>::buildGraph(const T * grid)
{
	const int & nc = grid->numCells();
	
	sdb::Sequence<sdb::Coord2> edgeMap;

static const int TwelveEdges[12][2] = {
{ 0, 1}, /// x
{ 2, 3},
{ 4, 5},
{ 6, 7},
{ 0, 2}, /// y
{ 1, 3},
{ 4, 6},
{ 5, 7},
{ 0, 4}, /// z
{ 1, 5},
{ 2, 6},
{ 3, 7}
};

	for(int i=0;i<nc;++i) {
		const int * hexav = grid->cellVertices(i);
		for(int j=0;j<12;++j) {
			sdb::Coord2 k = sdb::Coord2(hexav[TwelveEdges[j][0] ],
										hexav[TwelveEdges[j][1] ] ).ordered();
			if(!edgeMap.findKey(k) ) {
				edgeMap.insert(k);
			}
		}
	}
	
    std::map<int, std::vector<int> > vvemap;
	
	int c = 0;
	edgeMap.begin();
	while(!edgeMap.end() ) {
	
		int v0 = edgeMap.key().x;
		vvemap[v0].push_back(c);
		
		int v1 = edgeMap.key().y;
		vvemap[v1].push_back(c);
		
		c++;
		edgeMap.next();
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
	int ne = edgeMap.size();
	int ni = edgeInds.size();
	BaseDistanceField::create(nv, ne, ni);
    
    extractGridPos(grid);
	extractEdges(&edgeMap);
	extractEdgeBegins(edgeBegins);
	extractEdgeIndices(edgeInds);
    
    vvemap.clear();
	edgeBegins.clear();
	edgeInds.clear();
	
    calculateEdgeLength();
    
}

template<typename T>
void HexagonDistanceField<T>::extractGridPos(const T * grid)
{   
    DistanceNode * dst = nodes();
    const int n = grid->numPoints();
    for(int i=0;i<n;++i) {
        dst[i].pos = grid->pos(i);
    }
}

template<typename T>
Vector3F HexagonDistanceField<T>::getCellNormal(const T * grid, const int & idx,
							const Vector3F & pcenter) const
{
	int cellVs[8];
	grid->getCellVertices(cellVs, idx);
	
	Vector3F cellNml(0.f, 0.f, 0.f);
	
	int c = 0;
	for(int i=0;i<8;++i) {
		
		const DistanceNode & d = nodes()[cellVs[i]];
		
		if(d.val > 0.f) {
			cellNml += d.pos - pcenter;
			c++;
		}
	}
	
	if(c>0) {
		cellNml /= (float)c;
	}
	return cellNml;
}

}

}
#endif
