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

#include <graph/BaseDistanceField.h>

namespace aphid {

namespace topo {

class GeodesicDistance : public BaseDistanceField {
	
	float m_maxDist;
	
public:
	GeodesicDistance();
	virtual ~GeodesicDistance();
	
	void buildTriangleGraph(const int& vertexCount,
				const float* vertexPos,
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

protected:

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
