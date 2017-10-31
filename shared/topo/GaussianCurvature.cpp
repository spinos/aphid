/*
 *  GaussianCurvature.cpp
 *  
 *
 *  Created by jian zhang on 10/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GaussianCurvature.h"
#include <geom/ConvexShape.h>

namespace aphid {

namespace topo {

GaussianCurvature::GaussianCurvature()
{}

GaussianCurvature::~GaussianCurvature()
{}

void GaussianCurvature::calcCurvatures(const int& vertexCount,
				const float* vertexPos,
				const float* vertexNml,
				const int& triangleCount,
				const int* triangleIndices)
{
    accumVertexAreas(vertexCount,
                    vertexPos,
				    triangleCount,
				    triangleIndices);
	tagEdgeFace(triangleCount, triangleIndices);
	m_vertexPos = (const Vector3F*)vertexPos;
	m_vertexNml = (const Vector3F*)vertexNml;
	m_triangleIndices = triangleIndices;
	find1RingNeighbors();
	m_K.reset(new float[vertexCount]);
    memset(m_K.get(), 0, vertexCount<<2);
    calcK();
	m_H.reset(new float[vertexCount]);
    memset(m_H.get(), 0, vertexCount<<2);
    calcH();
}

void GaussianCurvature::accumVertexAreas(const int& vertexCount,
				const float* vertexPos,
				const int& triangleCount,
				const int* triangleIndices)
{
    m_A.reset(new float[vertexCount]);
	memset(m_A.get(), 0, vertexCount<<2);
	
    const Vector3F* vp = (const Vector3F*)vertexPos;
    cvx::Triangle ftri;
    for(int i=0;i<triangleCount;++i) {
        const int* trii = &triangleIndices[i * 3];
        ftri.set(vp[trii[0]],
            vp[trii[1]],
            vp[trii[2]]);
        
        float a = ftri.calculateArea() * .333333f;
        m_A[trii[0]] += a;
        m_A[trii[1]] += a;
        m_A[trii[2]] += a;
    }
}

void GaussianCurvature::tagEdgeFace(const int& triangleCount,
				const int* triangleIndices)
{
    const int& ne = numEdges();
    m_edgeFace.reset(new int[ne<<1]);
    for(int i=0;i<ne*2;++i) {
        m_edgeFace[i] = -1;
    }
    for(int i=0;i<triangleCount;++i) {
        const int* trii = &triangleIndices[i * 3];
        
        int j = edgeIndex(trii[0], trii[1]);
        setEdgeFace(j, i);
        
        j = edgeIndex(trii[0], trii[2]);
        setEdgeFace(j, i);
        
        j = edgeIndex(trii[1], trii[2]);
        setEdgeFace(j, i);
    }
}

void GaussianCurvature::setEdgeFace(const int& ej, const int& fi)
{
    if(m_edgeFace[ej<<1] < 0)
        m_edgeFace[ej<<1] = fi;
    else
        m_edgeFace[(ej<<1)+1] = fi;
}

bool GaussianCurvature::isEdgeOnBoundary(const int& i) const
{ return m_edgeFace[(i<<1)+1] < 0; }

bool GaussianCurvature::isVertexOnBoundary(int& ej, const int& i) const
{
    const int& endj = edgeBegins()[i+1];
	int vj, j = edgeBegins()[i];
	for(;j<endj;++j) {
		
		ej = edgeIndices()[j];
		if(isEdgeOnBoundary(ej) ) {
		    return true; 
		}
		
	}
    return false;
}

void GaussianCurvature::find1RingNeighbors()
{
    m_Vj.reset(new int[numEdgeIndices()]);
    const int& nv = numNodes();
    for(int i=0;i<nv;++i) {
        find1RingNeighborV(i);
    }
    
}

void GaussianCurvature::find1RingNeighborV(const int& i)
{
    const int j0 = edgeBegins()[i];
    const int& endj = edgeBegins()[i+1];
	int cj = edgeIndices()[j0];
    if(isVertexOnBoundary(cj, i) ) {
        
    }
	
/// first vj
	int j = j0;
	m_Vj[j]= oppositeNodeIndex(i, cj);
	j++;
	
	while (j < endj) {
		m_Vj[j] = nextVetexToEdge(cj, i, j0, j);
		cj = edgeIndex(i, m_Vj[j]);
		j++;
	}
}

int GaussianCurvature::nextVetexToEdge(const int& k, const int& vi, const int& j0, const int& j1)
{
/// 1st face
	int fi = m_edgeFace[k<<1];
	int r = oppositeVertexOnFace(&m_triangleIndices[fi*3], vi, m_Vj[j1 - 1]);
	
	if(!isVjVisited(r, j0, j1) )
		return r;
	
/// 2nd face	
	fi = m_edgeFace[(k<<1) + 1];
	if(fi < 0)
		return r;
		
	r = oppositeVertexOnFace(&m_triangleIndices[fi*3], vi, m_Vj[j1 - 1]);
	
	return r;
}

bool GaussianCurvature::isVjVisited(const int& x, const int& j0, const int& j1)
{
	for(int i=j0;i<j1;++i) {
		if(m_Vj[i] == x)
			return true;
	}
	return false;
}

int GaussianCurvature::oppositeVertexOnFace(const int* tri, const int& v1, const int& v2)
{
	if(tri[0] != v1 && tri[0] != v2)
		return tri[0];
	if(tri[1] != v1 && tri[1] != v2)
		return tri[1];
	return tri[2];
}

void GaussianCurvature::calcK()
{
    const int& nv = numNodes();
	for(int i=0;i<nv;++i) {
        calcKi(i);
    }
}

void GaussianCurvature::calcKi(const int& i)
{
	int cj;
	if(isVertexOnBoundary(cj, i) )
		return;
		
	const Vector3F& pvi = m_vertexPos[i];
	
	float& ki = m_K[i];
	const int j0 = edgeBegins()[i];
    const int& endj = edgeBegins()[i+1];
	for(int j=j0;j<endj;++j) {
		int j1 = j + 1;
		if(j1 == endj)
			j1 = j0;
		
		Vector3F e1 = m_vertexPos[m_Vj[j]] - pvi;
		Vector3F e2 = m_vertexPos[m_Vj[j1]] - pvi;
		e1.normalize();
		e2.normalize();
		float alpha = acos(e1.dot(e2) );
		
		ki += alpha;
	}
	
	ki = (6.283185f - ki) / m_A[i];
}

void GaussianCurvature::calcH()
{
	const int& nv = numNodes();
	for(int i=0;i<nv;++i) {
        calcHi(i);
    }

}

void GaussianCurvature::calcHi(const int& i)
{
	float& hi = m_H[i];
	hi = 1.f;
	int cj;
	if(isVertexOnBoundary(cj, i) )
		return;
		
	const Vector3F& pvi = m_vertexPos[i];
	
	const int j0 = edgeBegins()[i];
    const int& endj = edgeBegins()[i+1];
	for(int j=j0;j<endj;++j) {
		int j1 = j + 1;
		if(j1 == endj)
			j1 = j0;
			
		Vector3F e1 = m_vertexPos[m_Vj[j] ] - pvi;
		
		hi += e1.length() * acos(m_vertexNml[m_Vj[j] ].dot(m_vertexNml[m_Vj[j1] ]) );
	}
	
	hi = (.25f * hi) / m_A[i];
}

const float& GaussianCurvature::vertexArea(const int& i) const
{ return m_A[i]; }

const float* GaussianCurvature::K() const
{ return m_K.get(); }
	
const float* GaussianCurvature::H() const
{ return m_H.get(); }

const Vector3F* GaussianCurvature::vertexPos() const
{ return m_vertexPos; }

float GaussianCurvature::curvatureChange(const int& v1, const int& v2) const
{
	float r = K()[v1]/H()[v1] - K()[v2]/H()[v2];
	if(r < 0.f)
		r = -r;
	return r;
}

void GaussianCurvature::getVij(int& nvj, const int* & vj, const int& i) const
{ 
	const int& j0 = edgeBegins()[i];
	nvj = edgeBegins()[i+1] - j0;
	vj = &m_Vj[j0];
}

void GaussianCurvature::colorEdgeByCurvature(float* edgePos, float* edgeCol, const int& i)
{
	const IDistanceEdge & eg = edges()[i];
	const int& v1 = eg.vi.x;
	const int& v2 = eg.vi.y;
	float r = curvatureChange(v1, v2);

	r *= .67f;
		
	edgeCol[0] = r;
	edgeCol[1] = 0.f;
	edgeCol[2] = 1.f - r;
	
	memcpy(&edgePos[0], &nodes()[v1].pos, 12);
	memcpy(&edgePos[3], &nodes()[v2].pos, 12);
}

}

}