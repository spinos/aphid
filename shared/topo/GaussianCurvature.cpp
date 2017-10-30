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
	find1RingNeighbors();
	
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
        
        float a = ftri.calculateArea();
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
		
		int k = edgeIndices()[j];

		if(isEdgeOnBoundary(k) ) {
		    ej = k;
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
/// vj by edge
    int j = edgeBegins()[i];
    const int& endj = edgeBegins()[i+1];
    for(;j<endj;++j) {
		
        int k = edgeIndices()[j];
        m_Vj[j]= oppositeNodeIndex(i, k);
		
	}

/// re-order
    int ej;
    if(isVertexOnBoundary(ej, i) ) {
        findNeighborOnBoundary(ej, i);
        return;
    }
/// closed 
/// first in ring
    
}

void GaussianCurvature::findNeighborOnBoundary(const int& ej, const int& i)
{
    
}

const float& GaussianCurvature::vertexArea(const int& i) const
{ return m_A[i]; }

}

}