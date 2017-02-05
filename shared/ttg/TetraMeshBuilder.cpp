/*
 *  TetraMeshBuilder.cpp
 *  
 *
 *  Created by zhang on 17-2-5.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "TetraMeshBuilder.h"

namespace aphid {

namespace ttg {

TetraMeshBuilder::TetraMeshBuilder()
{}

TetraMeshBuilder::~TetraMeshBuilder()
{}

void TetraMeshBuilder::mapNodeInCell(std::map<int, Vector3F > & dst,
                        BccCell3 * cell )
{
    cell->begin();
	while(!cell->end() ) {
        
        const BccNode3 * node = cell->value();
        dst[node->index] = node->pos;
        
        cell->next();
	}
}

void TetraMeshBuilder::clearTetra(std::vector<ITetrahedron *> & tets)
{
	std::vector<ITetrahedron *>::iterator it = tets.begin();
	for(;it!=tets.end();++it) {
		delete *it;
	}
	tets.clear();
    
}

}
}
