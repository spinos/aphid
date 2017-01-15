/*
 *  TriangleMeshClique.cpp
 *  
 *
 *  Created by jian zhang on 12/15/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "TriangleMeshCluster.h"
#include <geom/ATriangleMesh.h>
#include <ConvexShape.h>
#include <sdb/cache_diff.h>

namespace aphid {

TriangleMeshCluster::TriangleMeshCluster()
{}

TriangleMeshCluster::~TriangleMeshCluster()
{
	m_sites.clear();
	m_edges.clear();
}

void TriangleMeshCluster::create(const float * points,
				const int & numTriangleIndices,
				const int * triangleIndices)
{
	const int nt = numTriangleIndices/3;
	cvx::Triangle telm;
	for(int i=0;i<nt;++i) {
		const int * tri = &triangleIndices[i*3];
		
		telm.setP(Vector3F(&points[tri[0] * 3]), 0);
		telm.setP(Vector3F(&points[tri[1] * 3]), 1);
		telm.setP(Vector3F(&points[tri[2] * 3]), 2);
		
		const sdb::Coord3 vs = sdb::Coord3(tri[0], tri[1], tri[2]);
		
		SiteData * d = new SiteData();
		d->_center = telm.center();
		d->_normal = telm.calculateNormal();
		d->_idx = i;
		d->_clique = -1;
		d->_distance = 1e8f;
		m_sites.insert(vs.ordered(), d );
		
		const sdb::Coord3 vsod = vs.ordered();
		const sdb::Coord2 e0(vs.x, vs.y);
		addEdge(e0.ordered(), vsod, telm.P(0), telm.P(1) );
		const sdb::Coord2 e1(vs.y, vs.z);
		addEdge(e1.ordered(), vsod, telm.P(1), telm.P(2) );
		const sdb::Coord2 e2(vs.z, vs.x);
		addEdge(e2.ordered(), vsod, telm.P(2), telm.P(0) );
	}
}

int TriangleMeshCluster::numSites()
{ return m_sites.size(); }

void TriangleMeshCluster::addEdge(const sdb::Coord2 & c,
				const sdb::Coord3 & f,
				const Vector3F & p0,
				const Vector3F & p1)
{
	EdgeData * e = m_edges.find(c );
	if(e) {
		if(e->_face1.x > -1) {
			std::cout<<"\n WARNNING edge"<<c<<" connected to more than 2 faces";
		}
		e->_face1 = f;
		e->_center = (p0 + p1) * .5f;
		return;
	}
	e = new EdgeData;
	e->_face0 = f;
	e->_face1.x = -1;
	e->_center = (p0 + p1) * .5f;
	m_edges.insert(c, e);
}

void TriangleMeshCluster::extractCliques(int * dst)
{
	m_sites.begin();
	while(!m_sites.end() ) {
		const SiteData * d = m_sites.value();
		dst[d->_idx] = d->_clique;
		
		m_sites.next();
	}
}

void TriangleMeshCluster::assignToCliqueOrigin(const int * triangleInd,
						const int & iclique)
{
	sdb::Coord3 vs(triangleInd[0], triangleInd[1], triangleInd[2]);
	SiteData * d = m_sites.find(vs.ordered() );
	if(!d) {
		std::cout<<"\n ERROR TriangleMeshCluster assignToClique cannot find site"
				<<vs;
		return;
	}
	d->_clique = iclique;
/// origin of clique
	d->_distance = 0.f;
}

void TriangleMeshCluster::buildCliques(const float & maxDistance)
{
	m_distanceLimit = maxDistance;
	
	sdb::Sequence<sdb::Coord3 > origins;
	
	m_sites.begin();
	while(!m_sites.end() ) {
		SiteData * d = m_sites.value();

		if(d->_distance == 0.f) {
			origins.insert(m_sites.key() );
			
		}
		
		m_sites.next();
	}
	
	origins.begin();
	while(!origins.end() ) {
		expandCliqueFrom(origins.key() );
		
		origins.next();
	}
	
}

void TriangleMeshCluster::expandCliqueFrom(const sdb::Coord3 & coord)
{
	//std::cout<<"\n expand from "<<coord;
	sdb::Sequence<sdb::Coord3 > expanded[2];
	expanded[1].insert(coord);
	
	bool added = true;
	while(added) {
	
		//std::cout<<" expand nc "<<expanded[1].size();
		expandClique(expanded[0], expanded[1]);
		
		int diff = sdb::cache_diff<sdb::Sequence<sdb::Coord3 > >(expanded[1], expanded[0]);
		//std::cout<<" diff nc "<<diff;
		
		added = (diff > 0);
		
	}
}

void TriangleMeshCluster::expandClique(sdb::Sequence<sdb::Coord3 > & c,
										sdb::Sequence<sdb::Coord3 > & q)
{
	q.begin();
	while(!q.end() ) {
		const SiteData * d = m_sites.find(q.key() );

		const sdb::Coord3 & vs = q.key();
		
		const sdb::Coord2 e1(vs.x, vs.y);
		connectSiteOnEdge(c, e1, vs, d);
		
		const sdb::Coord2 e2(vs.y, vs.z);
		connectSiteOnEdge(c, e2, vs, d);
		
/// order last edge
		const sdb::Coord2 e3(vs.x, vs.z);
		connectSiteOnEdge(c, e3, vs, d);
		
		q.next();
	}
}

void TriangleMeshCluster::connectSiteOnEdge(sdb::Sequence<sdb::Coord3 > & result,
					const sdb::Coord2 & ec,
					const sdb::Coord3 & vs,
					const SiteData * src)
{	
	EdgeData * ed = m_edges.find(ec);
	if(!ed) {
		std::cout<<"\n ERROR cannot find edge"<<ec;
		return;
	}
	
/// face on the other side
	sdb::Coord3 f1c;
	if(ed->_face0 == vs) {
		f1c = ed->_face1;
	} else if(ed->_face1 == vs) {
		f1c = ed->_face0;
	} else {
		std::cout<<"\n ERROR cannot edge"<<ec<<" not connected to face"<<vs;
		return;
	}
	
	SiteData * dst = m_sites.find(f1c);
	if(!dst) {
		std::cout<<"\n ERROR cannot find face"<<f1c;
		return;
	}

	float geod;
	if(isSiteCloseTo(geod, dst, ed, src) ) {
		//std::cout<<" d "<<geod;
/// assign to clique
		dst->_clique = src->_clique;
		dst->_distance = geod;
/// add to expanded
		result.insert(f1c);
		
	}
}

bool TriangleMeshCluster::isSiteCloseTo(float & d,
						const SiteData * dst,
						const EdgeData * edge,
						const SiteData * src)
{
	if(dst->_distance == 0.f) {
		return false;
	}

	d = src->_center.distanceTo(edge->_center)
				+ edge->_center.distanceTo(dst->_center);
	d += d * (1.f - src->_normal.dot(dst->_normal) ); 
	d += src->_distance;
				
	if(d > m_distanceLimit) {
		return false;
	}
	
	return (d < dst->_distance);
}

}
