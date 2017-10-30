/*
 *  GaussianCurvature.h
 *  
 *  gaussian curvature K maximum bending (rate of change) of the surface
 *  mean curvature H minimum bending of tangent direction 
 *  using gauss-bonnet scheme
 *  K = (2PI - sigma alpha_i) / (A / 3)
 *  H = (1/4 sigma ||e_i||beta_i) / (A / 3) 
 *  alpha is angle between two successive edges e_i = vvi
 *  beta is angle between normals of two successive neighbor vertices
 *  vi is 1-ring neighbor of v
 *  A is the accumulated areas of triangles around v
 *
 *  reference A comparison of Gaussian and mean curvature estimation methods 
 *  on triangular meshes of range image data
 *
 *  Created by jian zhang on 10/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TOPO_GAUSSIAN_CURVATURE_H
#define APH_TOPO_GAUSSIAN_CURVATURE_H

#include <graph/BaseDistanceField.h>
#include <boost/scoped_array.hpp>

namespace aphid {

namespace topo {

class GaussianCurvature : public BaseDistanceField {
  
    boost::scoped_array<float> m_A;
    boost::scoped_array<int> m_Vj;
    boost::scoped_array<int> m_edgeFace;
    
public:
    GaussianCurvature();
    virtual ~GaussianCurvature();
    
protected:
    void calcCurvatures(const int& vertexCount,
				const float* vertexPos,
				const float* vertexNml,
				const int& triangleCount,
				const int* triangleIndices);
    
	const float& vertexArea(const int& i) const;
	
private:
    void accumVertexAreas(const int& vertexCount,
				const float* vertexPos,
				const int& triangleCount,
				const int* triangleIndices);
	void tagEdgeFace(const int& triangleCount,
				const int* triangleIndices);
/// fi-face to ej-th edge
	void setEdgeFace(const int& ej, const int& fi);
    bool isEdgeOnBoundary(const int& i) const;
/// ej to first edge on boundary connected to i-th vertex
    bool isVertexOnBoundary(int& ej, const int& i) const;
	void find1RingNeighbors();
    void find1RingNeighborV(const int& i);
    void findNeighborOnBoundary(const int& ej, const int& i);
    
};

}

}

#endif
