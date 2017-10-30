/*
 *  GeodesicDistance.h
 *  
 *
 *  Created by jian zhang on 10/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TOPO_GEODESIC_DISTANCE_H
#define APH_TOPO_GEODESIC_DISTANCE_H

#include "GaussianCurvature.h"

namespace aphid {

class Vector3F;

namespace topo {

class GeodesicDistance : public GaussianCurvature {
	
	float m_maxDist; 
	
public:
	GeodesicDistance();
	virtual ~GeodesicDistance();
	
	void buildTriangleGraph(const int& vertexCount,
				const float* vertexPos,
				const float* vertexNml,
				const int& triangleCount,
				const int* triangleIndices);
				
	template<typename T>
	void setNodeValues(const T& nodeIndices, const float& val);
	void calaculateDistance(float* dest);

/// set node A zero	
/// for each node, distance to node A,
/// store node distances in dest
	void calaculateDistanceTo(float* dest, 
					const int& nodeA);
	
	const float& maxDistance() const;
	
protected:

	int getLowestNeightInd(const float* vals, const int& i);

protected:
/// average of all incident edges of i-th vertex
	float getAverageEdgeLength(const int& i) const;
/// reference mesh segmentation guided by seed points
/// (1 + |lp - lq|) ||np - nq||
	float getArcLen(const float* lqs,
			const Vector3F* nmls,
			const int& vp, const int& vq) const;
/// log(1 + arcos(angle_between(np, nq) ) / pi )
	float getAngleDistance(const Vector3F* nmls,
			const int& vp, const int& vq) const;
/// distance of every pair of vertices
/// arc lenght + angle distance
	void calcEdgeDistance(const float* vertexNml);
	
};

template<typename T>
void GeodesicDistance::setNodeValues(const T& nodeIndices, const float& val)
{
	resetNodes(1e20f, sdf::StBackGround, sdf::StUnknown);
	uncutEdges();
	
	const int n = nodeIndices.size();
	for(int i=0;i<n;++i) {
		DistanceNode & d = nodes()[nodeIndices[i]];
		d.val = 0.f;
		d.stat = sdf::StKnown;
	}
}

}

}

#endif
