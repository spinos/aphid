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

FitBccMeshBuilder::FitBccMeshBuilder() 
{
    m_samples = 0;
    m_reducedP = 0;
    m_octahedronSize = 0;
    m_octa = 0;
}

FitBccMeshBuilder::~FitBccMeshBuilder() 
{ cleanup(); }

void FitBccMeshBuilder::cleanup()
{
    if(m_samples) delete[] m_samples;
    if(m_reducedP) delete[] m_reducedP;
    if(m_octahedronSize) delete[] m_octahedronSize;
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
    cleanup();
	const unsigned ns = curve->numSegments();
	
	float * sl = new float[ns];
	float suml = 0.f;
	BezierSpline spl;
	
	unsigned i=0;
	for(;i<ns;i++) {
		curve->getSegmentSpline(i, spl);
		sl[i] = splineLength(spl);
		suml += sl[i];
	}
	
	// std::cout<<" total length "<<suml<<"\n";
	
	const float threashold = suml / (float)ns * 0.01f; 
	float shortestl = 1e8f;
	for(i=0;i<ns;i++) {
		if(sl[i]<shortestl && sl[i] > threashold) shortestl = sl[i];
	}
	
	// std::cout<<" shortest seg length "<<shortestl<<"\n";
	
	if(groupNCvRatio < .01f) groupNCvRatio = .01f;
	m_numGroups = ns * groupNCvRatio;
	
	if(m_numGroups < minNumGroups) m_numGroups = minNumGroups;
	if(m_numGroups > maxNumGroups) m_numGroups = maxNumGroups;
	// std::cout<<" split to "<<m_numGroups<<" groups\n";
	
	float splitL = suml / (float)ns /(float)m_numGroups * .249f;
	// std::cout<<" split length "<<splitL<<"\n";
	// if(splitL < suml / (float)ns * .06249f) splitL = suml / (float)ns * .06249f;
	
	unsigned nsplit = 0;
	for(i=0;i<ns;i++) {
		nsplit += sl[i] / splitL;
	}
	
	// std::cout<<" n split "<<nsplit<<"\n";
	
	m_samples = new Vector3F[nsplit + 2];
	
	unsigned isample = 0;
	float delta, param, curL;
	unsigned j;
	for(i=0;i<ns;i++) {
		curve->getSegmentSpline(i, spl);
		
		nsplit = sl[i] / splitL;
		delta = sl[i] / (float)nsplit;
		
		m_samples[isample] = spl.cv[0];
		isample++;
		
		curL = delta;
		
		param = splineParameterByLength(spl, curL);
		
		while(param < .99f) {
			m_samples[isample] = spl.calculateBezierPoint(param);
			isample++;
			
			curL += delta;
			param = splineParameterByLength(spl, curL);
		}
	}
	
	m_samples[isample] = curve->m_cvs[ns];
	
	m_numSamples = isample + 1;
	
	// std::cout<<" n sample "<<m_numSamples<<"\n";
	
	m_reducedP = new Vector3F[m_numGroups];
	unsigned * counts = new unsigned[m_numGroups];
	
	for(i=0; i<m_numGroups; i++) {
		m_reducedP[i].setZero();
		counts[i] = 0;
	}
	
	float fcpg = (float)m_numSamples / (float)m_numGroups;
	unsigned cpg = fcpg;
	if(fcpg - cpg > .5f) cpg++;
	
	// std::cout<<" n groups "<<m_numGroups<<" sample per group "<<cpg<<"\n";
	
	unsigned igrp;
	for(i=0; i<m_numSamples; i++) {
		igrp = i / cpg;
		if(igrp > m_numGroups-1) igrp = m_numGroups -1;
		m_reducedP[igrp] += m_samples[i];
		counts[igrp]++;
	}
	
	for(i=0; i<m_numGroups; i++) {
		m_reducedP[i] *= 1.f/(float)counts[i];
		// if(i==m_numGroups-1) std::cout<<" count in group"<<i<<" "<<counts[i]<<"\n";
	}
	
	delete[] counts;
	delete[] sl;
	
	m_octahedronSize = new float[m_numGroups];
	
	for(i=0; i<m_numGroups; i++) {
		if(i<1) {
			m_octahedronSize[i] = m_reducedP[0].distanceTo(m_reducedP[1]) * .5f;
		}
		else if(i==m_numGroups-1) {
			m_octahedronSize[i] = m_reducedP[m_numGroups-1].distanceTo(m_reducedP[m_numGroups-2]) * .5f;
		}
		else {
			m_octahedronSize[i] = m_reducedP[i].distanceTo(m_reducedP[i-1]) * .25f 
									+ m_reducedP[i].distanceTo(m_reducedP[i+1]) * .25f;

		}
		// std::cout<<" group size"<<i<<" "<<m_octahedronSize[i];
	}
	
	float averageSize = 0.f;
	for(i=0; i<m_numGroups; i++) averageSize += m_octahedronSize[i];
	averageSize /= (float)m_numGroups;
	// std::cout<<" average group size "<<averageSize<<"\n";
	
	int vv[2];
	int ee[2];
	float dV, dE;
	Vector3F a, b, c, d;
	
	m_octa = new BccOctahedron[m_numGroups];
	for(i=0; i<m_numGroups;i++) {
		m_octa[i].create(m_reducedP[i], m_octahedronSize[i]);
		
		if(i>0) {
			dV = m_octa[i].movePoleCost(vv, m_octa[i-1]);
			
			dE = m_octa[i].moveEdgeCost(ee, m_octa[i-1]);
			
			if(dV <= dE) {
				BccOctahedron::movePoles(m_octa[i], vv[0], m_octa[i-1], vv[1], tetrahedronP);
			}
			else {
				BccOctahedron::moveEdges(m_octa[i], ee[0], m_octa[i-1], ee[1], tetrahedronP);
			}
		}
		
		m_octa[i].createTetrahedron(tetrahedronP, tetrahedronInd);
		
		if(i>0) {
		    if(dV <= dE) {
				BccOctahedron::add8GapTetrahedron(m_octa[i], vv[0], m_octa[i-1], vv[1], 
													tetrahedronInd);
			}
			else {
				BccOctahedron::add2GapTetrahedron(m_octa[i], ee[0], m_octa[i-1], ee[1],
													tetrahedronInd);
			}
		}
	}
}

float FitBccMeshBuilder::splineLength(BezierSpline & spline)
{
	float res = 0.f;
	
	BezierSpline stack[64];
	int stackSize = 2;
	spline.deCasteljauSplit(stack[0], stack[1]);
	
	while(stackSize > 0) {
		BezierSpline c = stack[stackSize - 1];
		stackSize--;
		
		float l = c.cv[0].distanceTo(c.cv[3]);
		
		if(l < 1e-6f) {
			res += l;
			continue;
		}
		
		if(c.straightEnough()) {
			res += l;
			continue;
		}
			
		if(stackSize == 61) {
			std::cout<<" warning: FitBccMeshBuilder::splineLength stack overflown\n";
			continue;
		}
		
		BezierSpline a, b;
		c.deCasteljauSplit(a, b);
		
		stack[ stackSize ] = a;
		stackSize++;
		stack[ stackSize ] = b;
		stackSize++;
	}
	
	return res;
}

float FitBccMeshBuilder::splineParameterByLength(BezierSpline & spline, float expectedLength)
{
	float pmin = 0.f;
	float pmax = 1.f;
	float result = (pmin + pmax) * .5f;
	float lastResult = result;
	BezierSpline a, b, c;
	spline.deCasteljauSplit(a, b);
	
	float l = splineLength(a);
	while(Absolute(l - expectedLength) > 1e-4) {
		
		if(l > expectedLength) {
			c = a;
			c.deCasteljauSplit(a, b);
			
			pmax = result;
			
			l -= splineLength(b);
		}
		else {
			c = b;
			c.deCasteljauSplit(a, b);
			
			l += splineLength(a);
			
			pmin = result;
		}
		
		result = (pmin + pmax) * .5f;
		
		if(Absolute(result - lastResult) < 1e-4) break;
		
		lastResult = result;
	}
	return result;
}

