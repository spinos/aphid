/*
 *  FitBccMeshBuilder.cpp
 *  testbcc
 *
 *  Created by jian zhang on 4/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "FitBccMeshBuilder.h"
#include "BccGlobal.h"
#include <BezierCurve.h>
#include "BccOctahedron.h"
#include "bcc_common.h"
#include <KdTreeDrawer.h>
#include "CurveSampler.h"
#include "SampleGroup.h"
#define DBG_PRINT

float FitBccMeshBuilder::EstimatedGroupSize = 1.f;

FitBccMeshBuilder::FitBccMeshBuilder() 
{
    m_reducedP = 0;
    m_octa = 0;
	m_sampler = new CurveSampler;
	m_reducer = new SampleGroup;
}

FitBccMeshBuilder::~FitBccMeshBuilder() 
{ 
	cleanup(); 
	delete m_sampler;
	delete m_reducer;
}

void FitBccMeshBuilder::cleanup()
{
    if(m_reducedP) delete[] m_reducedP;
    if(m_octa) delete[] m_octa;
}
    
void FitBccMeshBuilder::build(GeometryArray * curves, 
	           std::vector<Vector3F > & tetrahedronP, 
	           std::vector<unsigned > & tetrahedronInd,
	           float groupNCvRatio,
			   unsigned minNumGroups,
	           unsigned maxNumGroups)
{
    const unsigned n = curves->numGeometries();
    unsigned i=0;
    for(;i<n;i++)
        build((BezierCurve *)curves->geometry(i), 
	           tetrahedronP, 
	           tetrahedronInd,
	           groupNCvRatio,
			   minNumGroups,
	           maxNumGroups);
}

void FitBccMeshBuilder::build(BezierCurve * curve, 
	           std::vector<Vector3F > & tetrahedronP, 
	           std::vector<unsigned > & tetrahedronInd,
	           float groupNCvRatio,
			   unsigned minNumGroups,
	           unsigned maxNumGroups)
{
	const unsigned lastNumTet = tetrahedronInd.size();
    cleanup();
	
	m_sampler->begin();
	m_sampler->process(curve, EstimatedGroupSize);
	m_sampler->end();
	
	const unsigned ns = curve->numSegments();
	
	float * sl = new float[ns];
	float suml = 0.f;
	BezierSpline spl;
	
	unsigned i=0;
	for(;i<ns;i++) {
		curve->getSegmentSpline(i, spl);
		sl[i] = BezierCurve::splineLength(spl);
		suml += sl[i];
	}

#ifdef DBG_PRINT
	std::cout<<" total length "<<suml<<"\n"
	<<" estimate n groups "<<1+(int)(suml/EstimatedGroupSize);
#endif

	const float threashold = suml / (float)ns * 0.01f;
	float shortestl = 1e8f;
	for(i=0;i<ns;i++) {
		if(sl[i]<shortestl && sl[i] > threashold) shortestl = sl[i];
	}
	
	// std::cout<<" shortest seg length "<<shortestl<<"\n";
	
	if(groupNCvRatio < .01f) groupNCvRatio = .01f;
	m_numGroups = (ns+1) * groupNCvRatio;
	
	if(m_numGroups < minNumGroups) m_numGroups = minNumGroups;
	if(m_numGroups > maxNumGroups) m_numGroups = maxNumGroups;
	// std::cout<<" split to "<<m_numGroups<<" groups\n";

	m_reducedP = new Vector3F[m_numGroups];
	unsigned * counts = new unsigned[m_numGroups];
	
	for(i=0; i<m_numGroups; i++) {
		m_reducedP[i].setZero();
		counts[i] = 0;
	}
	
	const unsigned numSamples = m_sampler->numSamples();
	Vector3F * samples = m_sampler->samples();
	
	m_reducer->compute(samples, numSamples, m_numGroups);
	
	float fcpg = (float)numSamples / (float)m_numGroups;
	unsigned cpg = fcpg;
	if(fcpg - cpg > .5f) cpg++;
#ifdef DBG_PRINT
	std::cout<<" n groups "<<m_numGroups<<" sample per group "<<cpg<<"\n";
#endif	
	unsigned igrp;
	for(i=0; i<numSamples; i++) {
		igrp = i / cpg;
		if(igrp > m_numGroups-1) igrp = m_numGroups -1;
		m_reducedP[igrp] += samples[i];
		counts[igrp]++;
	}
	
	for(i=0; i<m_numGroups; i++) {
		m_reducedP[i] *= 1.f/(float)counts[i];
		// if(i==m_numGroups-1) std::cout<<" count in group"<<i<<" "<<counts[i]<<"\n";
	}
	
	delete[] counts;
	delete[] sl;
	
	float * groupSize = new float[m_numGroups];
	for(i=0; i<m_numGroups;i++) groupSize[i] = -1e8f;
	
	for(i=0; i<numSamples; i++) {
		igrp = i / cpg;
		if(igrp > m_numGroups-1) igrp = m_numGroups -1;
		float s = m_reducedP[igrp].distanceTo(samples[i]);
		if(s>groupSize[igrp]) groupSize[igrp] = s;
	}
	
	int vv[2];
	int ee[2];
	float dV, dE;
	Vector3F octDir, a, b, c, d;
	
	m_octa = new BccOctahedron[m_numGroups];
	for(i=0; i<m_numGroups;i++) {
		if(i<1) {
			octDir = (m_reducedP[1] - m_reducedP[0]) * .5f;
		}
		else if(i==m_numGroups-1) {
			octDir = (m_reducedP[m_numGroups-1] - m_reducedP[m_numGroups-2]) * .5f;
		}
		else {
			octDir = (m_reducedP[i] - m_reducedP[i-1]) * .25f 
					+ (m_reducedP[i+1] - m_reducedP[i]) * .25f;

		}
		
		m_octa[i].create(m_reducedP[i], octDir, groupSize[i]);
		
		if(i>0) {
			dV = m_octa[i].movePoleCost(vv, m_octa[i-1]);
			
			dE = m_octa[i].moveEdgeCost(ee, m_octa[i-1]);
			
			BccOctahedron::moveEdges(m_octa[i], ee[0], m_octa[i-1], ee[1], tetrahedronP);
			
			BccOctahedron::connectDifferentAxis(m_octa[i], 
												m_octa[i-1], tetrahedronP);
		}
		
		m_octa[i].createTetrahedron(tetrahedronP, tetrahedronInd);
		
		if(i>0) {
			BccOctahedron::add2GapTetrahedron(m_octa[i], ee[0], m_octa[i-1], ee[1],
													tetrahedronInd);
		}
	}
	
	delete[] groupSize;
	
	checkTetrahedronVolume(tetrahedronP, tetrahedronInd, lastNumTet);
}

void FitBccMeshBuilder::drawOctahedron(KdTreeDrawer * drawer)
{
	for(unsigned i=0; i<m_numGroups; i++)
		drawOctahedron(drawer, m_octa[i]);
}

void FitBccMeshBuilder::drawOctahedron(KdTreeDrawer * drawer, BccOctahedron & octa)
{
	Vector3F a, b;
	glColor3f(0.f, 0.8f, 0.f);

	drawer->arrow(octa.p()[0], octa.p()[1]);
	
	glColor3f(0.8f, 0.f, 0.f);
	
	int i;
	for(i=0;i<8;i++) {
		octa.getEdge(a, b, i);
		drawer->arrow(a, b);
	}
	
	glColor3f(0.f, 0.f, 0.8f);
	
	for(i=8;i<12;i++) {
		octa.getEdge(a, b, i);
		drawer->arrow(a, b);
	}
}

void FitBccMeshBuilder::checkTetrahedronVolume(std::vector<Vector3F > & tetrahedronP, 
	           std::vector<unsigned > & tetrahedronInd, unsigned start)
{
	Vector3F p[4];
	unsigned i = start;
	unsigned tmp;
	unsigned tend = tetrahedronInd.size();
	for(;i<tend;i+=4) {
		p[0] = tetrahedronP[tetrahedronInd[i]];
		p[1] = tetrahedronP[tetrahedronInd[i+1]];
		p[2] = tetrahedronP[tetrahedronInd[i+2]];
		p[3] = tetrahedronP[tetrahedronInd[i+3]];
		// std::cout<<" tet vol "<<tetrahedronVolume(p)<<"\n";
		if(tetrahedronVolume(p)<0.f) {
			tmp = tetrahedronInd[i+1];
			tetrahedronInd[i+1] = tetrahedronInd[i+2];
			tetrahedronInd[i+2] = tmp;
			
			p[0] = tetrahedronP[tetrahedronInd[i]];
			p[1] = tetrahedronP[tetrahedronInd[i+1]];
			p[2] = tetrahedronP[tetrahedronInd[i+2]];
			p[3] = tetrahedronP[tetrahedronInd[i+3]];
			
			// std::cout<<" tet vol after swap 1 2 "<<tetrahedronVolume(p)<<"\n";
		}
	}
}

void FitBccMeshBuilder::drawSamples(KdTreeDrawer * drawer)
{
	const unsigned numSamples = m_sampler->numSamples();
	Vector3F * samples = m_sampler->samples();
	
	glDisable(GL_DEPTH_TEST);
	glColor3f(1.f, 0.f, 0.f);
	glBegin(GL_POINTS);
	unsigned i = 0;
	for(;i<numSamples;i++) glVertex3fv((float *)&samples[i]);
	glEnd();
	
	glColor3f(0.f, 1.f, 1.f);
	for(i=0;i<m_reducer->K()-1;i++) drawer->arrow(m_reducer->centeroid(i), m_reducer->centeroid(i+1));
	
}
