/*
 *  GaussianCurvature.h
 *  
 *  gaussian curvature K = k1k2 product of the principal curvatures
 *  mean curvature H = 1/2 (k1 + k2) average of principal curvatures
 *  maximum bending (rate of change) of the surface
 *  minimum bending of tangent direction 
 *  using gauss-bonnet scheme
 *  K = (2PI - sigma alpha_i) / (A / 3)
 *  absolute mean curvature |H| = (1/4 sigma ||e_i||beta_i) / (A / 3) 
 *  alpha is angle between two successive edges e_i = vvi
 *  beta is angle between normals of two successive neighbor vertices
 *  vi is 1-ring neighbor of v
 *  A is the accumulated areas of triangles around v
 *
 *  reference A comparison of Gaussian and mean curvature estimation methods 
 *  on triangular meshes of range image data
 *  Optimizing 3D Triangulations Using Discrete Curvature Analysis
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
  
    const Vector3F* m_vertexPos;
	const Vector3F* m_vertexNml;
	const int* m_triangleIndices;
	boost::scoped_array<float> m_A;
	boost::scoped_array<float> m_K;
    boost::scoped_array<float> m_H;
    boost::scoped_array<int> m_Vj;
    boost::scoped_array<int> m_edgeFace;
	
public:
    GaussianCurvature();
    virtual ~GaussianCurvature();
	
	void getVij(int& nvj, const int* & vj, const int& i) const;
    void colorEdgeByCurvature(float* edgePos, float* edgeCol, const int& i);
	
protected:
    void calcCurvatures(const int& vertexCount,
				const float* vertexPos,
				const float* vertexNml,
				const int& triangleCount,
				const int* triangleIndices);
    
	const float& vertexArea(const int& i) const;
	const float* K() const;
	const float* H() const;
/// cross v1 and v2
	float curvatureChange(const int& v1, const int& v2) const;
	
	const Vector3F* vertexPos() const;

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
/// ej is index to first edge on boundary connected to i-th vertex
    bool isVertexOnBoundary(int& ej, const int& i) const;
	void find1RingNeighbors();
    void find1RingNeighborV(const int& i);
/// opposite vertex on face connected to k-th edge
/// excluding vi, vj[j] j0 <= j < j1
	int nextVetexToEdge(const int& k, const int& vi, const int& j0, const int& j1);
	int oppositeVertexOnFace(const int* tri, const int& v1, const int& v2);
	bool isVjVisited(const int& x, const int& j0, const int& j1);
	void calcK();
	void calcKi(const int& i);
	void calcH();
	void calcHi(const int& i);

};

}

}

#endif
