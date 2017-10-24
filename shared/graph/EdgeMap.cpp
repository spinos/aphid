/*
 *  EdgeMap.cpp
 *  
 *
 *  Created by jian zhang on 10/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "EdgeMap.h"
#include <map>

namespace aphid {

namespace grh {

EdgeMap::EdgeMap()
{}

EdgeMap::~EdgeMap()
{ sdb::Array<sdb::Coord2, int>::clear(); }

void EdgeMap::createFromTriangles(const int& triangleCount,
				const int* triangleIndices)
{
	for(int i=0;i<triangleCount;++i) {
		const int* ti = &triangleIndices[i*3];
        addEdge(sdb::Coord2(ti[0], ti[1]).ordered() );
        addEdge(sdb::Coord2(ti[0], ti[2]).ordered() );
        addEdge(sdb::Coord2(ti[1], ti[2]).ordered() );
        
    }
}

void EdgeMap::addEdge(const sdb::Coord2 & e)
{
    if(find(e) ) {
        return;
    }
            
    int * rc = new int;
    *rc = -1;
    insert(e, rc);
}

int * EdgeMap::findEdge(const int & v1, const int & v2)
{
    return find(sdb::Coord2(v1, v2).ordered() );
}

void EdgeMap::resetIndices()
{
    begin();
    while(!end()) {
        *value() = -1;
        next();
    }
}

void EdgeMap::buildVertexVaryingEdges(std::vector<int>& edgeBegins,
				std::vector<int>& edgeInds)
{
	std::map<int, std::vector<int> > vvemap;
	
	int c = 0;
	begin();
	while(!end() ) {
	
		const int& v0 = key().x;
		vvemap[v0].push_back(c);
		
		const int& v1 = key().y;
		vvemap[v1].push_back(c);
		
		c++;
		next();
	}
	
	int nvve = 0;
	std::map<int, std::vector<int> >::iterator it = vvemap.begin();
	for(;it!=vvemap.end();++it) {
		edgeBegins.push_back(nvve);
		
		mergeIndices(it->second, edgeInds);
		
		nvve += (it->second).size();
		
		it->second.clear();
	}
	vvemap.clear();
	
}

void EdgeMap::mergeIndices(const std::vector<int> & a,
							std::vector<int> & b) const
{
	std::vector<int>::const_iterator it = a.begin();
	for(;it!=a.end();++it) {
		b.push_back(*it);
    }
}

}

}