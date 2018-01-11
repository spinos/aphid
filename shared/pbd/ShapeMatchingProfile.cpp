/*
 *  ShapeMatchingProfile.cpp
 *  
 *
 *  Created by jian zhang on 1/14/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "ShapeMatchingProfile.h"
#include <geom/ParallelTransport.h>
#include <math/Matrix33F.h>
#include <math/miscfuncs.h>
#include "ShapeMatchingRegion.h"

namespace aphid {
namespace pbd {

ShapeMatchingProfile::ShapeMatchingProfile() : m_numPoints(0),
m_numRegions(0)
{}

const Vector3F* ShapeMatchingProfile::x0() const
{ return m_x0.get(); }

const float* ShapeMatchingProfile::inverseMass() const
{ return m_invmass.get(); }

const int& ShapeMatchingProfile::numPoints() const
{ return m_numPoints; }

const int& ShapeMatchingProfile::numRegions() const
{ return m_numRegions; }

void ShapeMatchingProfile::createTestStrand()
{
	Vector3F g[32];
	for(int i=0;i<32;++i) {
		g[i].set(10.f + (.6f - .006 * i) * i, 10.f - .05f * i + 1.2f * (.5f + .005f * i) * sin(0.5f * i), 1.3f * (.5f + .005f * i) * cos(.5f * i) );
	}
	
	const int np = 64;
	const int hnp = np / 2;
	const float hw = .12f;
	
	m_x0.reset(new Vector3F[np]);
	m_invmass.reset(new float[np]);
	
	Vector3F p0p1 = g[1] - g[0];
	Matrix33F frm;
	ParallelTransport::FirstFrame(frm, p0p1, Vector3F(0.f, 1.f, 1.f) );
	Vector3F nml = ParallelTransport::FrameUp(frm);
	
	m_x0[0] = g[0] - nml * hw;
	m_x0[1] = g[0] + nml * hw;
	
	static const int hilb[2][2] = {{1, 0}, {0, 1}};
	
	Vector3F p1p2;
	for(int i=1;i<hnp;++i) {
		p1p2 = g[i+1] - g[i];
		ParallelTransport::RotateFrame(frm, p0p1, p1p2);
		nml = ParallelTransport::FrameUp(frm);
		
		m_x0[i * 2 + hilb[i&1][0] ] = g[i] + nml * hw;
		m_x0[i * 2 + hilb[i&1][1] ] = g[i] - nml * hw;
		
		p0p1 = p1p2;
	}
	
/// lock first segment
	for(int i=0;i<4;++i) {
		m_invmass[i] = 0.f;
	}
	
	for(int i=4;i<np;++i) {
		m_invmass[i] = 3.f;
	}
	
	m_numPoints = np;
	m_numRegions = hnp - 2;
	m_regionVertexBegin.reset(new int[m_numRegions + 1]);
	m_regionEdgeBegin.reset(new int[m_numRegions + 1]);
	
	int regionC = 0;
	int vertexC = 0;
	int edgeC = 0;

/// 0   3 - 4   7 ... 2n-2   2n+1 - 2n+2
/// |   |   |   |      |      |      |
/// 1 - 2   5 - 6     2n-1 - 2n     2n+3 ...

	for(int i=2;i<m_numPoints-4;i+=2) {
		m_regionVertexBegin[regionC] = vertexC;
		m_regionEdgeBegin[regionC] = edgeC;
/// 6 vertices per region		
		vertexC += 6;
/// 7 edges per region
		edgeC += 7;
		
		regionC++;
	}
	
	m_regionVertexBegin[regionC] = vertexC;
	m_regionEdgeBegin[regionC] = edgeC;
	
	m_vertices.reset(new int[vertexC]);
	m_edges.reset(new int[edgeC * 2]);
	
	regionC = 0;
	for(int i=2;i<m_numPoints-4;i+=2) {
		
		int* vr = &m_vertices[m_regionVertexBegin[regionC] ];
		
		vr[0] = i - 2;
		vr[1] = i - 1;
		vr[2] = i;
		vr[3] = i + 1;
		vr[4] = i + 2;
		vr[5] = i + 3;
		
		int* er = &m_edges[m_regionEdgeBegin[regionC] * 2 ];
		
		er[0] = i - 2;
		er[1] = i - 1;
		
		er[2] = i - 1;
		er[3] = i;
		
		er[4] = i;
		er[5] = i + 1;
		
		er[6] = i + 1;
		er[7] = i + 2;
		
		er[8] = i + 2;
		er[9] = i + 3;
		
		er[10] = i;
		er[11] = i + 3;
		
		er[12] = i + 1;
		er[13] = i - 2;
		
		regionC++;
	}
	
}

void ShapeMatchingProfile::getRegionVE(RegionVE& ve, const int& i) const
{
	ve._nv = m_regionVertexBegin[i+1] - m_regionVertexBegin[i];
	ve._ne = m_regionEdgeBegin[i+1] - m_regionEdgeBegin[i];
	ve._vinds = &m_vertices[m_regionVertexBegin[i] ];
	ve._einds = &m_edges[m_regionEdgeBegin[i] * 2 ];
}

}
}