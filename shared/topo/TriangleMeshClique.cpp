/*
 *  TriangleMeshClique.cpp
 *  
 *
 *  Created by jian zhang on 12/15/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "TriangleMeshClique.h"
#include <geom/ATriangleMesh.h>

namespace aphid {

TriangleMeshClique::TriangleMeshClique(const ATriangleMesh * mesh)
{
	m_mesh = mesh;
	const unsigned ntri = mesh->numTriangles();
	for(unsigned i=0;i<ntri;++i) {
		unsigned * v = mesh->triangleIndices(i);
		sdb::Coord3 vs(v[0], v[1], v[2]);
		
		SiteData * d = new SiteData();
		d->x = i;
		m_sites.insert(vs.ordered(), d );
	}
}

TriangleMeshClique::~TriangleMeshClique()
{
	m_sites.clear();
}

void TriangleMeshClique::findClique(const unsigned & idx)
{
	unsigned * v = m_mesh->triangleIndices(idx);
	sdb::Coord3 vs = sdb::Coord3(v[0], v[1], v[2]).ordered();
	
	sdb::Sequence<sdb::Coord3 > expanded[2];
	
	m_c.clear();
	
	int a = 0, b = 1;
	expanded[b].insert(vs);
	
	bool added = true;
	while(added) {
	
		expandClique(m_c, expanded[b]);
		
		bufferExpansion(m_c, expanded[a], expanded[b]);
		
		added = expanded[b].size() > 0;
		
	}
	
}

void TriangleMeshClique::expandClique(sdb::Sequence<sdb::Coord3 > & result,
									sdb::Sequence<sdb::Coord3 > & expanded)
{
	expanded.begin();
	while(!expanded.end()) {
	
		const sdb::Coord3 & vs = expanded.key();
		
		const sdb::Coord2 e1(vs.x, vs.y);
		findConnectedSites(result, e1);
		const sdb::Coord2 e2(vs.y, vs.z);
		findConnectedSites(result, e2);
		const sdb::Coord2 e3(vs.z, vs.x);
		findConnectedSites(result, e3);
	
		expanded.next();
	}
	
}

void TriangleMeshClique::findConnectedSites(sdb::Sequence<sdb::Coord3 > & result,
									const sdb::Coord2 & e)
{
	m_sites.begin();
	while(!m_sites.end() ) {
		
		const sdb::Coord3 & fv = m_sites.key();
		
		if(fv.x > e.y) {
			return;
		}
		
		if(fv.intersects(e) ) {
			result.insert(fv);
		}
		
		m_sites.next();
	}
}

void TriangleMeshClique::bufferExpansion(sdb::Sequence<sdb::Coord3 > & result,
					sdb::Sequence<sdb::Coord3 > & a,
					sdb::Sequence<sdb::Coord3 > & b)
{
	b.clear();
	
	result.begin();
	while(!result.end() ) {
		
		const sdb::Coord3 & k = result.key();
		if(!a.findKey(k ) ) {
			b.insert(k );
			a.insert(k );
		}
		result.next();
	}
			
}

void TriangleMeshClique::getCliqueSiteIndices(std::vector<int> & dst)
{
	m_c.begin();
	while(!m_c.end() ) {
		SiteData * d = m_sites.find(m_c.key() );
		if(!d) {
			throw "cannot find site";
		}
		dst.push_back(d->x );
		m_c.next();
	}
}

}
