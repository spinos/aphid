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
				const float* vertexNml,
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
	calcCurvatures(vertexCount,
				vertexPos, vertexNml,
				triangleCount, triangleIndices);
	calcEdgeDistance(vertexNml);
    
}

void GeodesicDistance::calcEdgeDistance(const float* vertexNml)
{
	const int & nv = numNodes();
	float* lqs = new float[nv];
	for(int i = 0;i<nv;++i) {
		lqs[i] = getAverageEdgeLength(i);
	}
	
	const Vector3F* nmls = (const Vector3F*)vertexNml;
	const int n = numEdges();
	for(int i=0;i<n;++i) {
		
		IDistanceEdge & ei = edges()[i];
		ei.len += 10.f * getArcLen(lqs, nmls, ei.vi.x, ei.vi.y);
		ei.len += 10.f * getAngleDistance(nmls, ei.vi.x, ei.vi.y);  
	}
	
	delete[] lqs;
}

float GeodesicDistance::getArcLen(const float* lqs,
			const Vector3F* nmls,
			const int& vp, const int& vq) const
{
	float t = lqs[vp] - lqs[vq];
	if(t < 0.f)
		t = -t;
	t += 1.f;
	return t * nmls[vp].distanceTo(nmls[vq]);
}

float GeodesicDistance::getAngleDistance(const Vector3F* nmls,
			const int& vp, const int& vq) const
{
	float t = nmls[vp].dot(nmls[vq]);
	t = acos(t);
	return log(1.f + t / 3.141593f);
}

float GeodesicDistance::getAverageEdgeLength(const int& i) const
{
	float r = 0.f;
	const int& endj = edgeBegins()[i+1];
	int j = edgeBegins()[i];
	const int c = endj - j;
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		r += eg.len;
		
	}
	return r / c;
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

int GeodesicDistance::getLowestNeightInd(const float* vals, const int& i)
{
	int r = -1;
	float minVal = 1e8f;
/// for each neighbor of A
	const int& endj = edgeBegins()[i+1];
	int vj, j = edgeBegins()[i];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == i) {
			vj = eg.vi.y;
        }
		
		if(minVal > vals[vj]) {
			minVal = vals[vj];
			r = vj;
		}
		
	}
	return r;
}

}

}