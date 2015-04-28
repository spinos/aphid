/*
 *  FitTest.cpp
 *  testbcc
 *
 *  Created by jian zhang on 4/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "FitTest.h"
#include "BccGlobal.h"
#include <KdTreeDrawer.h>
#include <BezierCurve.h>
#include <CurveBuilder.h>
#include "BccOctahedron.h"
#include "bcc_common.h"

FitTest::FitTest(KdTreeDrawer * drawer) 
{ 
	m_drawer = drawer; 
	m_curve = new BezierCurve;
	
	CurveBuilder cb;
	cb.addVertex(Vector3F(.5f, 1.f, 1.f));
	cb.addVertex(Vector3F(-2.f, 10.f, 6.01f));
	cb.addVertex(Vector3F(5.f, 10.f, 5.99f));
	cb.addVertex(Vector3F(5.f, 8.f, 4.01f));
	cb.addVertex(Vector3F(4.f, 8.f, 2.03f));
	cb.addVertex(Vector3F(4.f, 8.5f, 1.01f));
	cb.addVertex(Vector3F(3.1f, 8.95f, 0.f));
	
	cb.finishBuild(m_curve);
	
	unsigned ns = m_curve->numSegments();
	
	float * sl = new float[ns];
	float suml = 0.f;
	BezierSpline spl;
	
	unsigned i=0;
	for(;i<ns;i++) {
		m_curve->getSegmentSpline(i, spl);
		sl[i] = splineLength(spl);
		suml += sl[i];
	}
	
	std::cout<<" total length "<<suml<<"\n";
	
	const float threashold = suml / (float)ns * 0.01f; 
	float shortestl = 1e8f;
	for(i=0;i<ns;i++) {
		if(sl[i]<shortestl && sl[i] > threashold) shortestl = sl[i];
	}
	
	std::cout<<" shortest seg length "<<shortestl<<"\n";
	
	m_numGroups = 7;
	std::cout<<" split to "<<m_numGroups<<" groups\n";
	
	float splitL = suml / (float)ns /(float)m_numGroups * .249f;
	std::cout<<" split length "<<splitL<<"\n";
	// if(splitL < suml / (float)ns * .06249f) splitL = suml / (float)ns * .06249f;
	
	unsigned nsplit = 0;
	for(i=0;i<ns;i++) {
		nsplit += sl[i] / splitL;
	}
	
	std::cout<<" n split "<<nsplit<<"\n";
	
	m_samples = new Vector3F[nsplit + 2];
	
	unsigned isample = 0;
	float delta, param, curL;
	unsigned j;
	for(i=0;i<ns;i++) {
		m_curve->getSegmentSpline(i, spl);
		
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
	
	m_samples[isample] = m_curve->m_cvs[ns];
	
	m_numSamples = isample + 1;
	
	std::cout<<" n sample "<<m_numSamples<<"\n";
	
	m_reducedP = new Vector3F[m_numGroups];
	unsigned * counts = new unsigned[m_numGroups];
	
	for(i=0; i<m_numGroups; i++) {
		m_reducedP[i].setZero();
		counts[i] = 0;
	}
	
	float fcpg = (float)m_numSamples / (float)m_numGroups;
	unsigned cpg = fcpg;
	if(fcpg - cpg > .5f) cpg++;
	
	std::cout<<" n groups "<<m_numGroups<<" sample per group "<<cpg<<"\n";
	
	unsigned igrp;
	for(i=0; i<m_numSamples; i++) {
		igrp = i / cpg;
		if(igrp > m_numGroups-1) igrp = m_numGroups -1;
		m_reducedP[igrp] += m_samples[i];
		counts[igrp]++;
	}
	
	for(i=0; i<m_numGroups; i++) {
		m_reducedP[i] *= 1.f/(float)counts[i];
		if(i==m_numGroups-1) std::cout<<" count in group"<<i<<" "<<counts[i]<<"\n";
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
		std::cout<<" group size"<<i<<" "<<m_octahedronSize[i];
	}
	
	float averageSize = 0.f;
	for(i=0; i<m_numGroups; i++) averageSize += m_octahedronSize[i];
	averageSize /= (float)m_numGroups;
	std::cout<<" average group size "<<averageSize<<"\n";
	
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
				BccOctahedron::movePoles(m_octa[i], vv[0], m_octa[i-1], vv[1], m_tetrahedronP);
			}
			else {
				BccOctahedron::moveEdges(m_octa[i], ee[0], m_octa[i-1], ee[1], m_tetrahedronP);
			}
		}
		
		m_octa[i].createTetrahedron(m_tetrahedronP, m_tetrahedronInd);
		
		if(i>0) {
		    if(dV <= dE) {
				BccOctahedron::add8GapTetrahedron(m_octa[i], vv[0], m_octa[i-1], vv[1], 
													m_tetrahedronInd);
			}
			else {
				BccOctahedron::add2GapTetrahedron(m_octa[i], ee[0], m_octa[i-1], ee[1],
													m_tetrahedronInd);
			}
		}
	}
		
	std::cout<<" tetrahedron n p "<<m_tetrahedronP.size()<<"\n"
		<<" n tetrahedron "<<m_tetrahedronInd.size()/4<<"\n";
}

FitTest::~FitTest() {}

void FitTest::draw() 
{
	// m_drawer->linearCurve(*m_curve);
	// m_drawer->smoothCurve(*m_curve, 8);
	glBegin(GL_POINTS);
	unsigned i=0;
	
	glColor3f(0.f, 0.f, .5f);
	for(;i<m_numSamples;i++)
		glVertex3fv((GLfloat *)&m_samples[i]);
	glEnd();
	//glColor3f(0.8f, 0.f, 0.f);
	//for(i=1; i<m_numReducedP+1; i++)
		//m_drawer->arrow(m_reducedP[i-1], m_reducedP[i]);
	
	// int vv[2];
	// int ee[2];
	// float dV, dE;
	// Vector3F a, b, c, d;
	for(i=0; i<m_numGroups; i++) {
		drawOctahedron(m_octa[i]);
		/*if(i>0) {
			dV = m_octa[i].movePoleCost(vv, m_octa[i-1]);
			
			dE = m_octa[i].moveEdgeCost(ee, m_octa[i-1]);
			
			if(dV <= dE) {
				glColor3f(.5f, .5f, 0.f);
				m_drawer->arrow(m_octa[i].p()[vv[0]], m_octa[i-1].p()[vv[1]]);
			}
			else {
				glColor3f(0.f, .5f, .5f);
			
				m_octa[i].getEdge(a, b, ee[0]);
				m_octa[i-1].getEdge(c, d, ee[1]);
				m_drawer->arrow(a, c);
				m_drawer->arrow(b, d);
			}
		}*/
	}
	
	drawTetrahedron();
}

void FitTest::drawOctahedron(BccOctahedron & octa)
{
	Vector3F a, b;
	glColor3f(0.f, 0.8f, 0.f);

	m_drawer->arrow(octa.p()[0], octa.p()[1]);
	
	glColor3f(0.8f, 0.f, 0.f);
	
	int i;
	for(i=0;i<8;i++) {
		octa.getEdge(a, b, i);
		m_drawer->arrow(a, b);
	}
	
	glColor3f(0.f, 0.f, 0.8f);
	
	for(i=8;i<12;i++) {
		octa.getEdge(a, b, i);
		m_drawer->arrow(a, b);
	}
}


float FitTest::splineLength(BezierSpline & spline)
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
			std::cout<<" warning: fitTest::splineLength stack overflown\n";
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

float FitTest::splineParameterByLength(BezierSpline & spline, float expectedLength)
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

void FitTest::drawTetrahedron()
{
	glColor3f(.8f, .8f, .8f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_TRIANGLES);
    unsigned i, j;
    Vector3F q;
    unsigned tet[4];
	const unsigned nt = m_tetrahedronInd.size() / 4;
    for(i=0; i< nt; i++) {
        tet[0] = m_tetrahedronInd[i*4];
		tet[1] = m_tetrahedronInd[i*4+1];
		tet[2] = m_tetrahedronInd[i*4+2];
		tet[3] = m_tetrahedronInd[i*4+3];
        for(j=0; j< 12; j++) {
            q = m_tetrahedronP[ tet[ TetrahedronToTriangleVertex[j] ] ];
            glVertex3fv((GLfloat *)&q);
        }
    }
    glEnd();
}
