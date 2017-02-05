/*
 *  TetraMeshBuilder.cpp
 *  
 *
 *  Created by zhang on 17-2-5.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "TetraMeshBuilder.h"
#include <ttg/AdaptiveBccGrid3.h>
#include <ttg/tetrahedron_graph.h>
#include <geom/ConvexShape.h>

namespace aphid {

namespace ttg {

TetraMeshBuilder::TetraMeshBuilder()
{ m_pos = 0; }

TetraMeshBuilder::~TetraMeshBuilder()
{ clearTetra(); }

void TetraMeshBuilder::buildMesh(AdaptiveBccGrid3 * grid)
{
    clearTetra();
	
    grid->begin();
	while(!grid->end() ) {
/// undivided
		if(!grid->value()->hasChild() ) {
			grid->value()->connectTetrahedrons(m_tets, grid->key(), grid);
        }
			
		grid->next();
	}
    
    std::map<int, Vector3F > vertmap;
    grid->begin();
	while(!grid->end() ) {
        
        mapNodeInCell(vertmap, grid->value() );
        
        grid->next();
	}
    
    m_numVertices = vertmap.size();
    m_pos = new Vector3F[m_numVertices];
    std::map<int, Vector3F >::iterator it = vertmap.begin();
    for(int i=0;it!=vertmap.end();++it,++i) {
        m_pos[i] = it->second;
    }
    
    vertmap.clear();
    
}

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

void TetraMeshBuilder::clearTetra()
{
	std::vector<ITetrahedron *>::iterator it = m_tets.begin();
	for(;it!=m_tets.end();++it) {
		delete *it;
	}
	m_tets.clear();
    if(m_pos) {
        delete[] m_pos;
    }
}

int TetraMeshBuilder::numTetrahedrons() const
{ return m_tets.size(); }

const int & TetraMeshBuilder::numVertices() const
{ return m_numVertices; }

const Vector3F * TetraMeshBuilder::vertices() const
{ return m_pos; }

const ITetrahedron * TetraMeshBuilder::tetra(const int & i) const
{ return m_tets[i]; }

void TetraMeshBuilder::getTetra(cvx::Tetrahedron & tet,
                const int & i) const
{
    const ITetrahedron * c = tetra(i);
    tet.set(m_pos[c->iv0], 
            m_pos[c->iv1], 
            m_pos[c->iv2], 
            m_pos[c->iv3]);
}

}
}
