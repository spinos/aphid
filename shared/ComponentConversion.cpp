/*
 *  ComponentConversion.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 6/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "ComponentConversion.h"
#include <MeshTopology.h>
#include <VertexAdjacency.h>
#include <AllGeometry.h>

ComponentConversion::ComponentConversion() {}

void ComponentConversion::setTopology(MeshTopology * topo)
{
	m_topology = topo;
}

void ComponentConversion::facetToPolygon(const std::vector<unsigned> & src, std::vector<unsigned> & polyIds) const
{
	std::vector<unsigned>::const_iterator it;
	for(it = src.begin(); it != src.end(); ++it) {
		appendUnique(m_topology->getFacet(*it)->getPolygonIndex(), polyIds);
	}
}

void ComponentConversion::vertexToEdge(const std::vector<unsigned> & src, std::vector<unsigned> & dst) const
{
	if(src.size() < 2) return;
	
	std::vector<unsigned>::const_iterator it0;
	std::vector<unsigned>::const_iterator it1 = src.begin();
	it1++;
	for(; it1 != src.end(); ++it1) {
		it0 = it1;
		it0--;
		
		VertexAdjacency & adj = m_topology->getAdjacency(*it0);
		char found = 0;
		Edge * e = adj.connectedToVertexBy(*it1, found);
		if(found) {
			appendUnique(e->getIndex(), dst);
		}
	}
}

void ComponentConversion::vertexToPolygon(const std::vector<unsigned> & src, std::vector<unsigned> & polyIds, std::vector<unsigned> & vppIds) const
{
	if(src.size() < 2) return;
	
	std::vector<unsigned>::const_iterator it0;
	std::vector<unsigned>::const_iterator it1 = src.begin();
	it1++;
	for(; it1 != src.end(); ++it1) {
		it0 = it1;
		it0--;
		
		VertexAdjacency & adj = m_topology->getAdjacency(*it0);
		char found = 0;
		Edge * e = adj.connectedToVertexBy(*it1, found);
		if(found) {
			Facet *f = (Facet *)e->getFace();
			if(appendUnique(f->getPolygonIndex(), polyIds)) {
				vppIds.push_back(*it0);
				vppIds.push_back(*it1);
			}
			
			Edge *opp = e->getTwin();
			if(opp) {
				f = (Facet *)opp->getFace();
				if(appendUnique(f->getPolygonIndex(), polyIds)) {
					vppIds.push_back(*it0);
					vppIds.push_back(*it1);
				}
			}
		}
	}
}

char ComponentConversion::appendUnique(unsigned val, std::vector<unsigned> & dst) const
{
	std::vector<unsigned>::const_iterator it;
	for(it = dst.begin(); it != dst.end(); ++it) {
		if(*it == val)
			return 0;
	}
	
	dst.push_back(val);
	return 1;
}
