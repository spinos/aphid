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
#include <ttg/PointDistance.h>
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
    void calculateDistance(Tf * intersectF,
							CalcDistanceProfile & profile);
    
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

template<typename T>
template<typename Tf>
void TetrahedronDistanceField<T>::calculateDistance(Tf * intersectF,
							CalcDistanceProfile & profile)
{
#if 0
		std::cout<<"\n TetrahedronDistanceField::calculateDistance";
		std::cout.flush();
#endif
        resetNodes(1e20f, sdf::StBackGround, sdf::StUnknown);
        uncutEdges();
 
		float selR;
		BoundingBox selBx;
		PointDistance apnt;
		const int & np = numNodes();
        for(int i=0;i<np;++i) {
			apnt.setPos(nodes()[i].pos);
			selR = longestEdgeLength(i);
			selBx.set(apnt.pos(), selR);
			
			if(!intersectF->select(selBx) ) {
				continue;
			}
			
			apnt.compute(intersectF, selR, profile.snapDistance);
            
			if(!apnt.isValid() ) {
				continue;
			}
			
			setNodePosDistance(apnt.pos(), apnt.result(), i);
			
		}
        
/// propagate distance to all nodes        
        fastMarchingMethod();
        
        int iFar = nodeFarthestFrom(profile.referencePoint, profile.direction);
/// visit out nodes
        marchOutside(iFar);
/// unvisited nodes are inside
        setFarNodeInside();

#if 0
		std::cout<<"\n done.";
#endif
    }

}

}
#endif
