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

namespace aphid {

ComponentConversion::ComponentConversion() {}

void ComponentConversion::setTopology(MeshTopology * topo)
{
	m_topology = topo;
}

void ComponentConversion::edgeRing(const unsigned & src, std::vector<unsigned> & edgeIds) const
{
	Edge * e = m_topology->getEdge(src);
    
	Edge *para = m_topology->parallelEdge(e);
	if(!para) return;
	
	printf("para %i %i\n", para->v0()->getIndex(), para->v1()->getIndex());
	if(!appendUnique(para->getIndex(), edgeIds)) return;
		
	Edge * opp = para->getTwin();
	if(!opp) return;	
	for(unsigned i = 0; i < 50; i++) {
		para = m_topology->parallelEdge(opp);
		if(!para) return;
		
		printf("para %i %i\n", para->v0()->getIndex(), para->v1()->getIndex());
		if(!appendUnique(para->getIndex(), edgeIds)) return;
		
		opp = para->getTwin();
		if(!opp) return;
	}
}

void ComponentConversion::facetToPolygon(const std::vector<unsigned> & src, std::vector<unsigned> & polyIds) const
{
	std::vector<unsigned>::const_iterator it;
	for(it = src.begin(); it != src.end(); ++it) {
		appendUnique(m_topology->getFacet(*it)->getPolygonIndex(), polyIds);
	}
}

void ComponentConversion::edgeToVertex(const std::vector<unsigned> & src, std::vector<unsigned> & dst) const
{
	std::vector<unsigned>::const_iterator it = src.begin();
	for(; it != src.end(); ++it) {
		Edge * e = m_topology->getEdge(*it);
		appendUnique(e->v0()->getIndex(), dst);
		appendUnique(e->v1()->getIndex(), dst);
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
			Edge * opp = e->getTwin();
			if(opp)
				appendUnique(opp->getIndex(), dst);
		}
	}
}

void ComponentConversion::vertexToEdge(const std::vector<unsigned> & src, std::vector<Edge *> & dst) const
{
	std::vector<unsigned> edgeIds;
	vertexToEdge(src, edgeIds);

	std::vector<unsigned>::const_iterator it = edgeIds.begin();
	for(; it != edgeIds.end(); ++it) {
		dst.push_back(m_topology->getEdge(*it));
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

void ComponentConversion::edgeToPolygon(const std::vector<unsigned> & src, std::vector<unsigned> & polyIds, std::vector<unsigned> & vppIds) const
{
	std::vector<unsigned>::const_iterator it = src.begin();
	for(; it != src.end(); ++it) {
		Edge * e = m_topology->getEdge(*it);
		polyIds.push_back(((Facet *)e->getFace())->getPolygonIndex());
		vppIds.push_back(e->v0()->getIndex());
		vppIds.push_back(e->v1()->getIndex());
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

Vector3F ComponentConversion::vertexPosition(unsigned idx) const
{
	VertexAdjacency & adj = m_topology->getAdjacency(idx);
	return *adj.m_v;
}

}

