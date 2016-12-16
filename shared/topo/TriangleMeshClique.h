/*
 *  TriangleMeshClique.h
 *  
 *
 *  Created by jian zhang on 12/15/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  Clique c is a subset of sites in S that are all neightbors to one another
 */
#ifndef APH_TRIANGLE_MESH_CLIQUE_H
#define APH_TRIANGLE_MESH_CLIQUE_H

#include <sdb/Array.h>
#include <sdb/Types.h>

namespace aphid {

class ATriangleMesh;

class TriangleMeshClique {

	struct SiteData {
		int x;
	};
	
	sdb::Array<sdb::Coord3, SiteData > m_sites;
	sdb::Sequence<sdb::Coord3 > m_c;
	const ATriangleMesh * m_mesh;
	
public:
	TriangleMeshClique(const ATriangleMesh * mesh);
	virtual ~TriangleMeshClique();
	
/// given a site s by face idx, find c connected to s  
	void findClique(const unsigned & idx);
	void getCliqueSiteIndices(std::vector<int> & dst);
	
protected:

private:
	void expandClique(sdb::Sequence<sdb::Coord3 > & result,
					sdb::Sequence<sdb::Coord3 > & expanded);
/// add to b if result not in a
	void bufferExpansion(sdb::Sequence<sdb::Coord3 > & result,
					sdb::Sequence<sdb::Coord3 > & a,
					sdb::Sequence<sdb::Coord3 > & b);
	
/// add all sites contains e to clique
	void findConnectedSites(sdb::Sequence<sdb::Coord3 > & result,
					const sdb::Coord2 & e);
	
};
	
}
#endif