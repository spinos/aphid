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
m_numRegions(0), 
m_lod(1.f)
{}

void ShapeMatchingProfile::setLod(const float& x)
{ m_lod = x; }

const Vector3F* ShapeMatchingProfile::x0() const
{ return m_x0.get(); }

const float* ShapeMatchingProfile::inverseMass() const
{ return m_invmass.get(); }

const int& ShapeMatchingProfile::numPoints() const
{ return m_numPoints; }

const int& ShapeMatchingProfile::numRegions() const
{ return m_numRegions; }

const float& ShapeMatchingProfile::averageSegmentLength() const
{ return m_avgSegLen; }

float ShapeMatchingProfile::detailSize() const
{ return m_avgSegLen * m_lod; }

void ShapeMatchingProfile::clearStrands()
{ 
	m_strandX0.clear();
	m_strandBegin.clear();
	m_strandBegin.push_back(0);
	m_strandParams.clear();
}

void ShapeMatchingProfile::addStrandPoint(const Vector3F& x0)
{ m_strandX0.push_back(x0); }

void ShapeMatchingProfile::finishStrand(const StrandParam& param)
{ 
	m_strandBegin.push_back(m_strandX0.size() );
	m_strandParams.push_back(param);
}

void ShapeMatchingProfile::buildProfile()
{
	const int nstrand = m_strandBegin.size() - 1;
	std::cout<<"\n ShapeMatchingProfile build n strand "<<nstrand;
	
	float sumSegLen = 0.f;
	int numSeg = 0;
	for(int i=0;i<nstrand;++i) {
		addSegLen(sumSegLen, numSeg, i);
	}
	m_avgSegLen = sumSegLen / (float)numSeg;
	std::cout<<"\n n segment "<<numSeg
			<<"\n average segment length "<<m_avgSegLen;
	
	const int np = m_strandX0.size() * 2;
	
	m_x0.reset(new Vector3F[np]);
	m_invmass.reset(new float[np]);
	
	m_numPoints = np;
	std::cout<<"\n n point "<<m_numPoints;
	
	for(int i=0;i<nstrand;++i) {
		buildStrand(i);
	}
	
	int regionCount = 0;
	for(int i=0;i<nstrand;++i) {
		countRegions(regionCount, i);
	}
	m_numRegions = regionCount;
	std::cout<<"\n n region "<<m_numRegions;
	
	m_regionVertexBegin.reset(new int[m_numRegions + 1]);
	m_regionEdgeBegin.reset(new int[m_numRegions + 1]);
	
	regionCount = 0;
	int edgeCount = 0;
	int vertexCount = 0;
	for(int i=0;i<nstrand;++i) {
		countRegionEdges(regionCount, edgeCount, vertexCount, i);
	}
	
	m_regionVertexBegin[regionCount] = vertexCount;
	m_regionEdgeBegin[regionCount] = edgeCount;
	std::cout<<"\n n edge "<<edgeCount;
	
	m_vertices.reset(new int[vertexCount]);
	m_edges.reset(new int[edgeCount * 2]);
	
	regionCount = 0;
	for(int i=0;i<nstrand;++i) {
		buildRegions(regionCount, i);
	}
	std::cout<<"\n done";
	
}

void ShapeMatchingProfile::addSegLen(float& segLenSum, int& segCount, const int& i)
{
	const int& strandEnd = m_strandBegin[i+1];
		
	for(int j=m_strandBegin[i] + 1;j<strandEnd;++j) {
		Vector3F p0p1 = m_strandX0[j] - m_strandX0[j - 1];
		segLenSum += p0p1.length();
		segCount++;
	}
}

void ShapeMatchingProfile::buildStrand(const int& istrand)
{
	const int& low = m_strandBegin[istrand];
	const int& high = m_strandBegin[istrand + 1];
	const StrandParam& sparam = m_strandParams[istrand];
	const float hw = m_avgSegLen * .19f;
	
	Vector3F p0p1 = m_strandX0[low + 1] - m_strandX0[low];
	Matrix33F frm;
	ParallelTransport::FirstFrame(frm, p0p1, sparam._binormal );
	Vector3F nml = ParallelTransport::FrameUp(frm);
	
	m_x0[low * 2] = m_strandX0[low] - nml * hw;
	m_x0[low * 2 + 1] = m_strandX0[low] + nml * hw;
	
	static const int hilb[2][2] = {{1, 0}, {0, 1}};
	
	Vector3F p1p2;
	for(int i=low + 1;i<high;++i) {
		p1p2 = m_strandX0[i] - m_strandX0[i-1];
		ParallelTransport::RotateFrame(frm, p0p1, p1p2);
		nml = ParallelTransport::FrameUp(frm);
		
		m_x0[i * 2 + hilb[i&1][0] ] = m_strandX0[i] + nml * hw;
		m_x0[i * 2 + hilb[i&1][1] ] = m_strandX0[i] - nml * hw;
		
		p0p1 = p1p2;
	}
	
/// lock first segment
	for(int i=low * 2;i<low * 2 +4;++i) {
		m_invmass[i] = 0.f;
	}
	
	const float dm = sparam._mass1 - sparam._mass0;
	const float l = 2.f * (high - low);
	float alpha;
	
	for(int i=low * 2+4;i<high * 2;++i) {
		alpha = (float)(i - low*2) / l;
		m_invmass[i] = 1.f / (sparam._mass0 + dm * alpha);
	}	
	
}

void ShapeMatchingProfile::countRegions(int& regionCount, const int& i)
{ 
	const int np = strandNumPoints(i);
	for(int i=0;i<np-5;i+=2) {
		regionCount++;
	}
}

void ShapeMatchingProfile::countRegionEdges(int& regionCount, int& edgeCount, int& vertexCount,
				const int& istrand)
{
	const int np = strandNumPoints(istrand);
	
	for(int i=0;i<np-5;i+=2) {
		m_regionVertexBegin[regionCount] = vertexCount;
		m_regionEdgeBegin[regionCount] = edgeCount;
		
		vertexCount += 6;
		edgeCount += 7;
		
		regionCount++;
	}
}

void ShapeMatchingProfile::buildRegions(int& regionCount, const int& istrand)
{
	const int np = strandNumPoints(istrand);
	const int offset = m_strandBegin[istrand] * 2;
	
/// 0   3 - 4   7 ...    2n-5 - 2n-4   2n-1
/// |   |   |   |         |      |      |
/// 1 - 2   5 - 6    ... 2n-6   2n-3 - 2n-2
	
	for(int i=0;i<np-5;i+=2) {
		
		int* vr = &m_vertices[m_regionVertexBegin[regionCount] ];
		
		vr[0] = offset + i;
		vr[1] = offset + i + 1;
		vr[2] = offset + i + 2;
		vr[3] = offset + i + 3;
		vr[4] = offset + i + 4;
		vr[5] = offset + i + 5;
		
		//std::cout<<"\n region "<<regionCount<<" "<<vr[0]<<","<<vr[1]<<","<<vr[2]
		//	<<","<<vr[3]<<","<<vr[4]<<","<<vr[5];
		
		int* er = &m_edges[m_regionEdgeBegin[regionCount] * 2 ];
		
		er[0] = offset + i;
		er[1] = offset + i + 1;
		
		er[2] = offset + i + 1;
		er[3] = offset + i + 2;
		
		er[4] = offset + i + 2;
		er[5] = offset + i + 3;
		
		er[6] = offset + i + 3;
		er[7] = offset + i + 4;
		
		er[8] = offset + i + 4;
		er[9] = offset + i + 5;
		
		er[10] = offset + i;
		er[11] = offset + i + 3;
		
		er[12] = offset + i + 2;
		er[13] = offset + i + 5;
		
		regionCount++;
	}
	
	// std::cout<<"\n end strand "<<(offset)<<","<<(offset + np);
}

int ShapeMatchingProfile::strandNumPoints(const int& i) const
{ return (m_strandBegin[i+1] - m_strandBegin[i] ) * 2; }

void ShapeMatchingProfile::getRegionVE(RegionVE& ve, const int& i) const
{
	ve._nv = m_regionVertexBegin[i+1] - m_regionVertexBegin[i];
	ve._ne = m_regionEdgeBegin[i+1] - m_regionEdgeBegin[i];
	ve._vinds = &m_vertices[m_regionVertexBegin[i] ];
	ve._einds = &m_edges[m_regionEdgeBegin[i] * 2 ];
}

}
}