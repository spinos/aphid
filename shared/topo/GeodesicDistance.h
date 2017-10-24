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
	
/// for each node, distance to node A,
/// store node distances in dest
	void calaculateDistanceTo(float* dest,
				const int& nodeA);
	
	const float& maxDistance() const;
	
protected:

protected:

};

}

}

#endif
