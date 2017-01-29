/*
 *  TetraGridEdgeMap.h
 *  
 *
 *  Created by zhang on 17-1-29.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_TTG_TETRA_GRID_EDGE_MAP
#define APH_TTG_TETRA_GRID_EDGE_MAP

#include <sdb/Array.h>

namespace aphid {

template<typename T>
class TetraGridEdgeMap : public sdb::Array<sdb::Coord2, int> {

public:
    TetraGridEdgeMap(const T * grd);
    virtual ~TetraGridEdgeMap();
    
    int * findEdge(const int & v1, const int & v2);
    void resetIndices();
    
protected:

private:
    void addEdge(const sdb::Coord2 & e);
    
};

template<typename T>
TetraGridEdgeMap<T>::TetraGridEdgeMap(const T * grd)
{
    sdb::Coord2 k;
    const int & nt = grd->numCells();
    for(int i=0;i<nt;++i) {
        const sdb::Coord4 & c = grd->cellVertices(i);
        addEdge(sdb::Coord2(c.x, c.y).ordered() );
        addEdge(sdb::Coord2(c.y, c.z).ordered() );
        addEdge(sdb::Coord2(c.z, c.x).ordered() );
        addEdge(sdb::Coord2(c.x, c.w).ordered() );
        addEdge(sdb::Coord2(c.y, c.w).ordered() );
        addEdge(sdb::Coord2(c.z, c.w).ordered() );
    }
}

template<typename T>
TetraGridEdgeMap<T>::~TetraGridEdgeMap()
{ sdb::Array<sdb::Coord2, int>::clear(); }

template<typename T>
void TetraGridEdgeMap<T>::addEdge(const sdb::Coord2 & e)
{
    if(find(e) ) {
        return;
    }
            
    int * rc = new int;
    *rc = -1;
    insert(e, rc);
}

template<typename T>
int * TetraGridEdgeMap<T>::findEdge(const int & v1, const int & v2)
{
    return find(sdb::Coord2(v1, v2).ordered() );
}

template<typename T>
void TetraGridEdgeMap<T>::resetIndices()
{
    begin();
    while(!end()) {
        *value() = -1;
        next();
    }
}

}
#endif
