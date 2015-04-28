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

static const int OctahedronEdge89Pair[4][2] = {
{8,8},{8,9},
{9,8},{9,9}};

static const int OctahedronEdgeValenceVertices[8][2] = {
{3,4},
{5,2},
{2,5},
{4,3},
{4,3},
{2,5},
{5,2},
{3,4}};

BccOctahedron::BccOctahedron() {}
BccOctahedron::~BccOctahedron() {}
	
Vector3F * BccOctahedron::p() 
{ return m_p; }

int * BccOctahedron::vertexTag()
{ return m_tag; }

unsigned * BccOctahedron::vertexIndex()
{ return m_meshVerticesInd; }

const int BccOctahedron::axis() const
{ return m_axis; }

const float BccOctahedron::size() const
{ return m_size; }

void BccOctahedron::create(const Vector3F & center, const Vector3F & dir)
{
	float size = dir.length();
	m_axis = dir.longestAxis();
	if(m_axis!=1) m_axis= 0;
	if(m_axis == 0) {
		m_p[0] = center - Vector3F::YAxis * size;
		m_p[1] = center + Vector3F::YAxis * size;
		m_p[2] = center - Vector3F::XAxis * size  - Vector3F::ZAxis * size;
		m_p[3] = center - Vector3F::XAxis * size  + Vector3F::ZAxis * size;
		m_p[4] = center + Vector3F::XAxis * size  - Vector3F::ZAxis * size;
		m_p[5] = center + Vector3F::XAxis * size  + Vector3F::ZAxis * size;
	}
	else {
		m_p[0] = center - Vector3F::XAxis * size;
		m_p[1] = center + Vector3F::XAxis * size;
		m_p[2] = center - Vector3F::YAxis * size  - Vector3F::ZAxis * size;
		m_p[3] = center - Vector3F::YAxis * size  + Vector3F::ZAxis * size;
		m_p[4] = center + Vector3F::YAxis * size  - Vector3F::ZAxis * size;
		m_p[5] = center + Vector3F::YAxis * size  + Vector3F::ZAxis * size;
	}
	
	m_tag[0] = 1;
	m_tag[1] = 1;
	m_tag[2] = 1;
	m_tag[3] = 1;
	m_tag[4] = 1;
	m_tag[5] = 1;
	
	m_size = size;
}

void BccOctahedron::closestPoleTo(int & va, int & vb,  BccOctahedron & another)
{
	float minD = 1e8f;
	float d;
	Vector3F disp = Vector3F::XAxis * m_size * .13f;
	if(m_axis != 1) disp = Vector3F::YAxis * m_size * .13f;
	int i, j;
	for(i=0;i<2;i++) {
		for(j=0;j<2;j++) {
			d = (m_p[i] + disp).distanceTo(another.p()[j]);
			if(d<minD) {
				minD = d;
				va = i;
				vb = j;
			}
		}
	}
}

void BccOctahedron::closestEdge89To(int & ea, int & eb,  BccOctahedron & another, int pole)
{
	float minD = 1e8f;
	float dist;
	Vector3F dispA = Vector3F::XAxis * m_size * .31f;
	if(m_axis != 1) dispA = Vector3F::YAxis * m_size * .31f;
	
	Vector3F dispB = Vector3F::XAxis * m_size * .31f;
	if(another.axis() != 1) dispB = Vector3F::YAxis * another.size() * .31f;
	
	Vector3F a, b, c, d;
	int i;
	for(i=0;i<4;i++) {
		getEdge(a, b, OctahedronEdge89Pair[i][0]);
		another.getEdge(c, d, OctahedronEdge89Pair[i][1]);
		
		dist =  (a+dispA).distanceTo(c+dispB) + (b+dispA).distanceTo(d+dispB);
		if(dist<minD) {
			minD = dist;
			ea = OctahedronEdge89Pair[i][0];
			eb = OctahedronEdge89Pair[i][1];
		}
	}
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
	if(octa1.axis() != octa2.axis()) return;
	
	Vector3F dv = (octa1.p()[va] - octa2.p()[vb]) * .5f;
	octa1.p()[va] -= dv;
	octa2.p()[vb] += dv;
	
	octa1.vertexTag()[va] = 0;
	octa1.vertexIndex()[va] = octa2.vertexIndex()[vb];
	
	points[octa2.vertexIndex()[vb]] = octa2.p()[vb];
}

void BccOctahedron::moveEdges(BccOctahedron & octa1, int ea, BccOctahedron & octa2, int eb, std::vector<Vector3F > & points)
{
	if(octa1.axis() != octa2.axis()) return;
	
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

void BccOctahedron::add8GapTetrahedron(BccOctahedron & octa1, int va, 
	                               BccOctahedron & octa2, int vb,
	                               std::vector<unsigned > & indices)
{
	if(octa1.axis() != octa2.axis()) return;
	
    indices.push_back(octa1.vertexIndex()[2]);
    indices.push_back(octa1.vertexIndex()[3]);
    indices.push_back(octa2.vertexIndex()[2]);
    indices.push_back(octa1.vertexIndex()[va]);
    
    indices.push_back(octa2.vertexIndex()[2]);
    indices.push_back(octa1.vertexIndex()[3]);
    indices.push_back(octa2.vertexIndex()[3]);
    indices.push_back(octa1.vertexIndex()[va]);
    
    indices.push_back(octa1.vertexIndex()[2]);
    indices.push_back(octa2.vertexIndex()[2]);
    indices.push_back(octa1.vertexIndex()[4]);
    indices.push_back(octa1.vertexIndex()[va]);
    
    indices.push_back(octa1.vertexIndex()[4]);
    indices.push_back(octa2.vertexIndex()[2]);
    indices.push_back(octa2.vertexIndex()[4]);
    indices.push_back(octa1.vertexIndex()[va]);
    
    indices.push_back(octa1.vertexIndex()[4]);
    indices.push_back(octa2.vertexIndex()[4]);
    indices.push_back(octa1.vertexIndex()[5]);
    indices.push_back(octa1.vertexIndex()[va]);
    
    indices.push_back(octa1.vertexIndex()[5]);
    indices.push_back(octa2.vertexIndex()[4]);
    indices.push_back(octa2.vertexIndex()[5]);
    indices.push_back(octa1.vertexIndex()[va]);
    
    indices.push_back(octa1.vertexIndex()[5]);
    indices.push_back(octa2.vertexIndex()[3]);
    indices.push_back(octa1.vertexIndex()[3]);
    indices.push_back(octa1.vertexIndex()[va]);
    
    indices.push_back(octa1.vertexIndex()[5]);
    indices.push_back(octa2.vertexIndex()[5]);
    indices.push_back(octa2.vertexIndex()[3]);
    indices.push_back(octa1.vertexIndex()[va]);
}

void BccOctahedron::add2GapTetrahedron(BccOctahedron & octa1, int ea, 
	                               BccOctahedron & octa2, int eb,
	                               std::vector<unsigned > & indices)
{
	if(octa1.axis() != octa2.axis()) return;
	
    int va, vb, vc, vd;
    if(ea>7) {
        octa1.getEdgeVertices(va, vb, ea);
        indices.push_back(octa1.vertexIndex()[va]);
        indices.push_back(octa1.vertexIndex()[vb]);
        indices.push_back(octa1.vertexIndex()[0]);
        indices.push_back(octa2.vertexIndex()[0]);
        
        indices.push_back(octa1.vertexIndex()[vb]);
        indices.push_back(octa1.vertexIndex()[va]);
        indices.push_back(octa1.vertexIndex()[1]);
        indices.push_back(octa2.vertexIndex()[1]);
    }
    else {
        octa1.getEdgeVertices(va, vb, ea);
        vc = OctahedronEdgeValenceVertices[ea][0];
        vd = OctahedronEdgeValenceVertices[eb][0];
        
        indices.push_back(octa1.vertexIndex()[va]);
        indices.push_back(octa1.vertexIndex()[vb]);
        indices.push_back(octa1.vertexIndex()[vc]);
        indices.push_back(octa2.vertexIndex()[vd]);
        
        vc = OctahedronEdgeValenceVertices[ea][1];
        vd = OctahedronEdgeValenceVertices[eb][1];
        
        indices.push_back(octa1.vertexIndex()[vb]);
        indices.push_back(octa1.vertexIndex()[va]);
        indices.push_back(octa1.vertexIndex()[vc]);
        indices.push_back(octa2.vertexIndex()[vd]);
    }
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

void BccOctahedron::connectDifferentAxis(BccOctahedron & octa1,
										BccOctahedron & octa2,
										std::vector<Vector3F > & points)
{
	if(octa1.axis() == octa2.axis()) return;
	
	int va, vb, vc, vd;
	
	octa1.closestPoleTo(va, vb, octa2);
	
	Vector3F dv = (octa1.p()[va] - octa2.p()[vb]) * .5f;
	octa1.p()[va] -= dv;
	octa2.p()[vb] += dv;
	
	octa1.vertexTag()[va] = 0;
	octa1.vertexIndex()[va] = octa2.vertexIndex()[vb];
	
	points[octa2.vertexIndex()[vb]] = octa2.p()[vb];
	
	int vai = 0;
	if(va==0) vai= 1;
	
	// return;
	
	int ea, eb;
	octa1.closestEdge89To(ea, eb, octa2, vai);
	
	Vector3F a, b, c, d;
	octa1.getEdge(a, b, ea);
	octa2.getEdge(c, d, eb);
				
	Vector3F dv1 = (a - c) * .5f;
	Vector3F dv2 = (b - d) * .5f;
	
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
