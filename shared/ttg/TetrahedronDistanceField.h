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

}

}
#endif
