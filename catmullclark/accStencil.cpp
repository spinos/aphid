/*
 *  accStencil.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "accStencil.h"
#include <iostream>
#include <Vector3F.h>
#include <VertexAdjacency.h>
#include "Edge.h"
#include <PatchNeighborRec.h>

AccStencil::AccStencil() {}
AccStencil::~AccStencil() {}

void AccStencil::setVertexPosition(Vector3F* data)
{
	_positions = data;
}

void AccStencil::setVertexNormal(Vector3F* data)
{
	_normals = data;
}

void AccStencil::resetCorner(int i)
{
	AccCorner &topo = m_corners[i];
	const int centerIdx = m_patchVertices[i];
    topo.setCenterIndex(centerIdx);
    topo.setCenterPosition(&_positions[centerIdx]);
    topo.setCenterNormal(&_normals[centerIdx]);
	
	topo.reset();
}

void AccStencil::restoreCorners(int ci, PatchNeighborRec * rec)
{
	resetCorner(ci);
	AccCorner &topo = m_corners[ci];
	
	const unsigned ne = rec->numEdges();
	unsigned i;
	for(i = 0; i < ne; i++) topo.addEdgeNeighbor(rec->edges()[i], _positions, _normals);
		
	const unsigned nc = rec->numCorners();
	for(i = 0; i < nc; i++) {
		if(rec->tagCorners()[i] == 1) {
			topo.addCornerNeighbor(rec->corners()[i], _positions, _normals);
		}
		else {
			topo.addCornerNeighborBetween(rec->corners()[i], rec->corners()[i+1], _positions, _normals);
			i++;
		}
	}
}

void AccStencil::findCorner(int vi)
{
	resetCorner(vi);
	AccCorner &topo = m_corners[vi];
	const int centerIdx = m_patchVertices[vi];
    
    VertexAdjacency &adj = m_vertexAdjacency[centerIdx];
    
    std::vector<int> neis;
    VertexAdjacency::VertexNeighbor *neighbor;
    int neighborIdx;
    for(neighbor = adj.firstNeighbor(); !adj.isLastNeighbor(); neighbor = adj.nextNeighbor()) {
        neighborIdx = neighbor->v->getIndex();
        
        neis.push_back(neighborIdx);
    }
    
    Edge dummy;
    for(std::vector<int>::iterator it = neis.begin(); it != neis.end(); ++it) {
        adj.findEdge(adj.getIndex(), *it, dummy);
        if(dummy.isReal())
            topo.addEdgeNeighbor(*it, _positions, _normals);
        else
            topo.addCornerNeighbor(*it, _positions, _normals);
    }
    
	findFringeCornerNeighbors(centerIdx, topo);
}

void AccStencil::findFringeCornerNeighbors(int c, AccCorner & topo)
{
    Edge dummy;
    int fringeNei;
    for(int i=0; i < topo.valence(); i++) {
        int i1 = i + 1;
        i1 = i1 % topo.valence();
        
        int nei0 = topo.edgeIndex(i);
        int nei1 = topo.edgeIndex(i1);

        VertexAdjacency &adj0 = m_vertexAdjacency[nei0];
        
        if(adj0.findEdge(adj0.getIndex(), nei1, dummy)) {
            if(dummy.isReal())
                topo.addCornerNeighborBetween(nei0, nei1, _positions, _normals);
            else if(findSharedNeighbor(nei0, nei1, c, fringeNei))
				topo.addCornerNeighbor(fringeNei, _positions, _normals);
        }
    }
}

char AccStencil::findSharedNeighbor(int a, int b, int c, int & dst)
{
    VertexAdjacency &adjA = m_vertexAdjacency[a];
    VertexAdjacency &adjB = m_vertexAdjacency[b];
    VertexAdjacency::VertexNeighbor *neighborA;
    VertexAdjacency::VertexNeighbor *neighborB;
    
    for(neighborA = adjA.firstNeighbor(); !adjA.isLastNeighbor(); neighborA = adjA.nextNeighbor()) {
        dst = neighborA->v->getIndex();
        if(dst != c) {
            for(neighborB = adjB.firstNeighbor(); !adjB.isLastNeighbor(); neighborB = adjB.nextNeighbor()) {
                if(dst == neighborB->v->getIndex()) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

void AccStencil::findEdge(int vi)
{
	AccEdge &topo = m_edges[vi];
	int v1 = vi + 1;
	v1 = v1 % 4;
	int e0 = m_patchVertices[vi];
	int e1 = m_patchVertices[v1];
	
	AccCorner &corner0 = m_corners[vi];
	
	topo.reset();
	
	if(e1 == e0) {
		topo._isZeroLength = 1;
		topo._edgePositions[0] = corner0.computePosition();
		topo._edgeNormals[0] = corner0.computeNormal();
		return;
	}
	
	topo._edgePositions[0] = _positions[e0];
	topo._edgeNormals[0] = _normals[e0];
	topo._edgePositions[1] = _positions[e1];
	topo._edgeNormals[1] = _normals[e1];

	AccCorner &corner1 = m_corners[v1];
	
	if(corner0.isOnBoundary() && corner1.isOnBoundary()) {
		topo._isBoundary = 1;
		return;
	}
	
	int a, b;
	corner0.edgeNeighborBeside(e1, a, b);

	//printf("%i-%i e %i %i \n", e0, e1, a, b);

	topo._fringePositions[0] = _positions[a];
	topo._fringePositions[1] = _positions[b];
	topo._fringeNormals[0] = _normals[a];
	topo._fringeNormals[1] = _normals[b];

	corner1.edgeNeighborBeside(e0, a, b);

	//printf("%i-%i e %i %i \n", e1, e0, a, b);
	
	topo._fringePositions[2] = _positions[a];
	topo._fringePositions[3] = _positions[b];
	topo._fringeNormals[2] = _normals[a];
	topo._fringeNormals[3] = _normals[b];
	
	topo._valence[0] = corner0.valence();
	topo._valence[1] = corner1.valence();
}

void AccStencil::findInterior(int vi)
{
	AccInterior &topo = m_interiors[vi];
	topo._valence = m_corners[vi].valence();
	for(int i = 0; i < 4; i++) {
		int ii = vi + i;
		ii = ii % 4;
		ii = m_patchVertices[ii];
		topo._cornerPositions[i] = _positions[ii];
		topo._cornerNormals[i] = _normals[ii];
	}
}

void AccStencil::verbose() const
{
}

