/*
 *  GeodesicDistance.cpp
 *
 *  Created by jian zhang on 10/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GeodesicDistance.h"
#include <graph/EdgeMap.h>

namespace aphid {

namespace topo {

GeodesicDistance::GeodesicDistance()
{}

GeodesicDistance::~GeodesicDistance()
{}

void GeodesicDistance::buildTriangleGraph(const int& vertexCount,
				const float* vertexPos,
				const int& triangleCount,
				const int* triangleIndices)
{
	grh::EdgeMap emap;
	emap.createFromTriangles(triangleCount, triangleIndices);
	
	std::vector<int> edgeBegins;
	std::vector<int> edgeInds;
	emap.buildVertexVaryingEdges(edgeBegins, edgeInds);
	
	int ne = emap.size();
	int ni = edgeInds.size();
	BaseDistanceField::create(vertexCount, ne, ni);
    
	DistanceNode * dst = nodes();
    const Vector3F* pv = (const Vector3F*)vertexPos;
    for(int i=0;i<vertexCount;++i) {
        dst[i].pos = pv[i];
    }
	
	extractEdges(&emap);
	extractEdgeBegins(edgeBegins);
	extractEdgeIndices(edgeInds);
    
	edgeBegins.clear();
	edgeInds.clear();
	
	calculateEdgeLength();
    
}

void GeodesicDistance::calaculateDistance(float* dest)
{
	fastMarchingMethod();
	
	m_maxDist = 0.f;
	const int & n = numNodes();
	for(int i = 0;i<n;++i) {
		DistanceNode & d = nodes()[i];
        dest[i] = d.val;
		
		if(m_maxDist < d.val && d.val < 1e19f )
			m_maxDist = d.val;
	}
}

void GeodesicDistance::calaculateDistanceTo(float* dest, 
								const int& nodeA)
{
	resetNodes(1e20f, sdf::StBackGround, sdf::StUnknown);
	uncutEdges();
	
	DistanceNode & d = nodes()[nodeA];
	d.val = 0.f;
	d.stat = sdf::StKnown;
	
	calaculateDistance(dest);
}

const float& GeodesicDistance::maxDistance() const
{ return m_maxDist; }

}

}