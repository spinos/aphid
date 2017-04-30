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
#include <math/Vector3F.h>

namespace aphid {

class ATriangleMesh;

class TriangleMeshClique {

	struct SiteData {
		int x;
	};
	
	sdb::Array<sdb::Coord3, SiteData > m_sites;
	sdb::Sequence<sdb::Coord3 > m_c;
	const ATriangleMesh * m_mesh;
	int m_cSize;
	
public:
	TriangleMeshClique(const ATriangleMesh * mesh);
	virtual ~TriangleMeshClique();
	
/// given a site s by idx, find c connected to s
/// limit site count  
	bool findClique(const unsigned & idx,
					const int & maxSiteCount);
	void getCliqueSiteIndices(std::vector<int> & dst);
	void getCliqueSiteIndices(sdb::Sequence<int> & dst);
/// extract positions of vertices connected to s in c
	void getCliqueVertexPositions(std::vector<Vector3F> & dst);
	const int & numSites() const;
	
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