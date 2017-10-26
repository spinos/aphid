/*
 *  GeodesicRegion.h
 *  
 *  mesh segmentation by region growth
 *
 *  Created by jian zhang on 10/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TOPO_GEODESIC_REGION_H
#define APH_TOPO_GEODESIC_REGION_H

#include "GeodesicDistance.h"
#include "DistancePath.h"

namespace aphid {

namespace topo {

class GeodesicRegion : public GeodesicDistance, public DistancePath {

public:
	GeodesicRegion();
	virtual ~GeodesicRegion();
	
	void createFromTriangles(const int& vertexCount,
				const float* vertexPos,
				const float* vertexNml,
				const int& triangleCount,
				const int* triangleIndices);
/// maximize sum of distance from root to all seeds
	bool findRootNode();
/// propagate from root and seed points
	void growRegions();
	
protected:
	
private:
/// from known nodes
/// add unlabeled neighbors  to trails
/// label each trail same as known neighbor with lowest distance
	void propagateLabels();
	
	void propagateLabels(std::deque<int > & heap, 
						const int & i);
												
};

}

}

#endif
