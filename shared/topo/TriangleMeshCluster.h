/*
 *  TriangleMeshCluster.h
 *  
 *
 *  Created by jian zhang on 12/15/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_TRIANGLE_MESH_CLUSTER_H
#define APH_TRIANGLE_MESH_CLUSTER_H

#include <sdb/Array.h>
#include <sdb/Types.h>

namespace aphid {

class TriangleMeshCluster {

	struct SiteData {
		Vector3F _center;
		Vector3F _normal;
		float _distance;
		int _idx;
		int _clique;
	};
	
	sdb::Array<sdb::Coord3, SiteData > m_sites;
	
	struct EdgeData {
		sdb::Coord3 _face0;
		sdb::Coord3 _face1;
		Vector3F _center;
	};
	
	sdb::Array<sdb::Coord2, EdgeData > m_edges;
	float m_distanceLimit;
	
public:
	TriangleMeshCluster();
	virtual ~TriangleMeshCluster();
	
	void create(const float * points,
				const int & numTriangleIndices,
				const int * triangleIndices);
				
	void assignToClique(const int * triangleInd,
						const int & iclique);
	
	int numSites();
	
/// expand from origin within certain distance 
	void buildCliques(const float & maxDistance);
	
/// dump the result
	void extractCliques(int * dst);
	
protected:

private:
	void addEdge(const sdb::Coord2 & c,
				const sdb::Coord3 & f,
				const Vector3F & p0,
				const Vector3F & p1);
	void expandCliqueFrom(const sdb::Coord3 & coord);
	void expandClique(sdb::Sequence<sdb::Coord3 > & c,
					sdb::Sequence<sdb::Coord3 > & q);
	void findConnectedSites(sdb::Sequence<sdb::Coord3 > & result,
					const sdb::Coord2 & ec,
					const sdb::Coord3 & vs,
					const SiteData * src);
	bool isSiteCloseTo(const SiteData * dst,
					const EdgeData * edge,
					const SiteData * src);
	
};
	
}
#endif