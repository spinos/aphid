/*
 *  BccOctahedron.cpp
 *  
 *
 *  Created by jian zhang on 4/28/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "BccOctahedron.h"
#include <bcc_common.h>

static const int OctahedronEdge[12][2] = {
{0,2}, {0,3}, {0,4}, {0,5},
{2,1}, {3,1}, {4,1}, {5,1},
{2,3}, {4,5}, {2,4}, {3,5}};

static const int OctahedronPolePair[2][2] = {
{0,1}, {1,0}};

static const int OctahedronEdgePair[12][2] = {
{0,7},{7,0},
{1,6},{6,1},
{2,5},{5,2},
{3,4},{4,3},
{8,9},{9,8},
{10,11},{11,10}};

BccOctahedron::BccOctahedron() {}
BccOctahedron::~BccOctahedron() {}
	
Vector3F * BccOctahedron::p() 
{ return m_p; }

int * BccOctahedron::vertexTag()
{ return m_tag; }

unsigned * BccOctahedron::vertexIndex()
{ return m_meshVerticesInd; }

void BccOctahedron::create(const Vector3F & center, float size)
{
	m_p[0] = center - Vector3F::XAxis * size;
	m_p[1] = center + Vector3F::XAxis * size;
	m_p[2] = center - Vector3F::YAxis * size  - Vector3F::ZAxis * size;
	m_p[3] = center - Vector3F::YAxis * size  + Vector3F::ZAxis * size;
	m_p[4] = center + Vector3F::YAxis * size  - Vector3F::ZAxis * size;
	m_p[5] = center + Vector3F::YAxis * size  + Vector3F::ZAxis * size;
	
	m_tag[0] = 1;
	m_tag[1] = 1;
	m_tag[2] = 1;
	m_tag[3] = 1;
	m_tag[4] = 1;
	m_tag[5] = 1;
}

float BccOctahedron::movePoleCost(int * v, BccOctahedron & another)
{
	float minD = 1e8f;
	float d;
	int i;
	for(i=0;i<2;i++) {
		d = m_p[OctahedronPolePair[i][0]].distanceTo(another.p()[OctahedronPolePair[i][1]]);
		if(d<minD) {
			minD = d;
			v[0] = OctahedronPolePair[i][0];
			v[1] = OctahedronPolePair[i][1];
		}
	}
	return minD * 2.f + 1e-6f;
}

float BccOctahedron::moveEdgeCost(int * v, BccOctahedron & another)
{
	float minD = 1e8f;
	float dist;
	int i;
	Vector3F a, b, c, d;
	for(i=0;i<12;i++) {
		getEdge(a, b, OctahedronEdgePair[i][0]);
		another.getEdge(c, d, OctahedronEdgePair[i][1]);
		
		dist = a.distanceTo(c) + b.distanceTo(d);
		if(dist<minD) {
			minD = dist;
			v[0] = OctahedronEdgePair[i][0];
			v[1] = OctahedronEdgePair[i][1];
		}
	}
	return minD;
}

void BccOctahedron::getEdge(Vector3F & a, Vector3F & b, int idx)
{
	a = m_p[OctahedronEdge[idx][0]];
	b = m_p[OctahedronEdge[idx][1]];
}

void BccOctahedron::getEdgeVertices(int & a, int & b, int idx)
{
	a = OctahedronEdge[idx][0];
	b = OctahedronEdge[idx][1];
}

void BccOctahedron::movePoles(BccOctahedron & octa1, int va, BccOctahedron & octa2, int vb, std::vector<Vector3F > & points)
{
	Vector3F dv = (octa1.p()[va] - octa2.p()[vb]) * .5f;
	octa1.p()[va] -= dv;
	octa2.p()[vb] += dv;
	
	octa1.vertexTag()[va] = 0;
	octa1.vertexIndex()[va] = octa2.vertexIndex()[vb];
	
	points[octa2.vertexIndex()[vb]] = octa2.p()[vb];
}

void BccOctahedron::moveEdges(BccOctahedron & octa1, int ea, BccOctahedron & octa2, int eb, std::vector<Vector3F > & points)
{
	Vector3F a, b, c, d;
	octa1.getEdge(a, b, ea);
	octa2.getEdge(c, d, eb);
				
	Vector3F dv1 = (a - c) * .5f;
	Vector3F dv2 = (b - d) * .5f;
	
	int va, vb, vc, vd;
	octa1.getEdgeVertices(va, vb, ea);
	octa1.p()[va] -= dv1;
	octa1.p()[vb] -= dv2;
	
	octa2.getEdgeVertices(vc, vd, eb);
	octa2.p()[vc] += dv1;
	octa2.p()[vd] += dv2;
	
	octa1.vertexTag()[va] = 0;
	octa1.vertexIndex()[va] = octa2.vertexIndex()[vc];
	
	octa1.vertexTag()[vb] = 0;
	octa1.vertexIndex()[vb] = octa2.vertexIndex()[vd];
	
	points[octa2.vertexIndex()[vc]] = octa2.p()[vc];
	points[octa2.vertexIndex()[vd]] = octa2.p()[vd];
}

void BccOctahedron::createTetrahedron(std::vector<Vector3F > & points, std::vector<unsigned > & indices)
{
	int i, j;
	for(i=0;i<6;i++) {
		if(m_tag[i]>0) {
			points.push_back(m_p[i]);
			m_meshVerticesInd[i] = points.size() - 1;
		}
	}
	
	// for(i=0;i<6;i++) std::cout<<" v octa "<<i<<" "<<m_meshVerticesInd[i]<<" ";
	
	for(i=0;i<4;i++) {
		for(j=0;j<4;j++) {
			indices.push_back(m_meshVerticesInd[OctahedronToTetrahedronVetex[i][j]]);
		}
	}
}
