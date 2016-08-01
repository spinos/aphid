/*
 *  AdaptiveBccField.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaptiveBccField.h"
#include <iostream>

using namespace aphid;

namespace ttg {

AdaptiveBccField::AdaptiveBccField()
{ m_insideOutsidePref.set(.577e6f, .577e6f, .577e6f); }

AdaptiveBccField::~AdaptiveBccField()
{}

void AdaptiveBccField::setInsideOutsidePref(const Vector3F & q)
{ m_insideOutsidePref = q; }

void AdaptiveBccField::buildGraph()
{
	sdb::Sequence<sdb::Coord2> egs;
	
	int v0, v1, v2, v3;
	const int n = numTetrahedrons();
	int i = 0;
	for(; i<n; ++i) {
		const ITetrahedron * t = tetra(i);
		v0 = t->iv0;
		v1 = t->iv1;
		v2 = t->iv2;
		v3 = t->iv3;
		egs.insert(sdb::Coord2(v0, v1).ordered() );
		egs.insert(sdb::Coord2(v0, v2).ordered() );
		egs.insert(sdb::Coord2(v0, v3).ordered() );
		egs.insert(sdb::Coord2(v1, v2).ordered() );
		egs.insert(sdb::Coord2(v1, v3).ordered() );
		egs.insert(sdb::Coord2(v2, v3).ordered() );
	}
	
	std::map<int, std::vector<int> > vvemap;
	
	int c = 0;
	egs.begin();
	while(!egs.end() ) {
	
		int v0 = egs.key().x;
		vvemap[v0].push_back(c);
		
		int v1 = egs.key().y;
		vvemap[v1].push_back(c);
		
		c++;
		egs.next();
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
	
	int nv = numVertices();
	int ne = egs.size();
	int ni = edgeInds.size();
	ADistanceField::create(nv, ne, ni);
	
	extractGridNodes<AdaptiveBccGrid3, BccNode3 >(nodes(), grid() );
	extractEdges(&egs);
	extractEdgeBegins(edgeBegins);
	extractEdgeIndices(edgeInds);
	
	vvemap.clear();
	edgeBegins.clear();
	edgeInds.clear();
	egs.clear();
	
	calculateEdgeLength();
	
}

void AdaptiveBccField::pushIndices(const std::vector<int> & a,
							std::vector<int> & b)
{
	std::vector<int>::const_iterator it = a.begin();
	for(;it!=a.end();++it) 
		b.push_back(*it);
}

void AdaptiveBccField::verbose()
{
	ADistanceField::verbose();
	std::cout<<"\n grid n cell "<<grid()->size()
		<<"\n n tetra "<<numTetrahedrons()
		<<"\n estimated error (min/max) "<<minError()<<"/"<<maxError();
		
	std::cout.flush();
}

void AdaptiveBccField::getTetraShape(cvx::Tetrahedron & b, const int & i) const
{ 
	const ITetrahedron * a = tetra(i);
	b.set(nodes()[a->iv0].pos,
			nodes()[a->iv1].pos,
			nodes()[a->iv2].pos,
			nodes()[a->iv3].pos);
}

void AdaptiveBccField::markTetraOnFront(const int & i)
{
	const ITetrahedron * a = tetra(i);
	nodes()[a->iv0].label = sdf::StFront;
	nodes()[a->iv1].label = sdf::StFront;
	nodes()[a->iv2].label = sdf::StFront;
	nodes()[a->iv3].label = sdf::StFront;
	addDirtyEdge(a->iv0, a->iv1);
	addDirtyEdge(a->iv0, a->iv2);
	addDirtyEdge(a->iv0, a->iv3);
	addDirtyEdge(a->iv1, a->iv2);
	addDirtyEdge(a->iv1, a->iv3);
	addDirtyEdge(a->iv2, a->iv3);
}

void AdaptiveBccField::subdivideGridByError(const float & threshold,
							const int & level)
{
	std::vector<aphid::sdb::Coord4 > divided;
	
	const DistanceNode * v = nodes();
	AdaptiveBccGrid3 * g = grid();
	const float r = g->levelCellSize(level) * .12f;

	BoundingBox dirtyBx;
	
	sdb::Sequence<sdb::Coord2 > * egs = dirtyEdges();
	egs->begin();
	while(!egs->end() ) {
		
		const sdb::Coord2 & ei = egs->key();
		const IDistanceEdge * ae = edge(ei.x, ei.y );
		if(ae ) {
		
			if(reconstructError(ae) > threshold ) {
			
			const Vector3F & p1 = v[ei.x].pos;
			const Vector3F & p2 = v[ei.y].pos;
			
			dirtyBx.reset();
			dirtyBx.expandBy(p1, r);
			dirtyBx.expandBy(p2, r);
			
			g->subdivideToLevel(dirtyBx, level+1, &divided);
		}
		}
		
		egs->next();
	}
	
	enforceBoundary(divided);
	
	divided.clear();

}

int AdaptiveBccField::findFarInd()
{
	BccCell3 * cell = grid()->cellClosestTo(m_insideOutsidePref);
	cell->begin();
	while(!cell->end() ) {
	
		if(isNodeBackground(cell->value()->index) ) {
			std::cout<<"\n far node pos"<<cell->value()->pos;
			return cell->value()->index;
		}
		cell->next();
	}
	return 0;
}

const float & AdaptiveBccField::errorThreshold() const
{ return m_errorThreshold; }

void AdaptiveBccField::buildGrid()
{ grid()->build(this); }

}