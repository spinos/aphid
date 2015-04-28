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
#include "GeometryArray.h"
#include <RandomCurve.h>
#include <bezierPatch.h>

FitTest::FitTest(KdTreeDrawer * drawer) 
{ 
	m_drawer = drawer;
	m_allGeo = new GeometryArray;
	
	// createSingleCurve();
	createRandomCurves();

	build(m_allGeo, m_tetrahedronP, m_tetrahedronInd, .87f, 4, 39);
	
	std::cout<<" tetrahedron n p "<<m_tetrahedronP.size()<<"\n"
		<<" n tetrahedron "<<m_tetrahedronInd.size()/4<<"\n done\n";
}

FitTest::~FitTest() {}

void FitTest::createSingleCurve()
{
    m_allGeo->create(1);
    
    BezierCurve * curve = new BezierCurve;
	
	CurveBuilder cb;
	cb.addVertex(Vector3F(.5f, 1.f, 1.f));
	cb.addVertex(Vector3F(-2.f, 10.f, 6.01f));
	cb.addVertex(Vector3F(5.f, 10.f, 5.99f));
	cb.addVertex(Vector3F(5.f, 8.f, 4.01f));
	cb.addVertex(Vector3F(4.f, 8.f, 2.03f));
	cb.addVertex(Vector3F(4.f, 8.5f, 1.01f));
	cb.addVertex(Vector3F(3.1f, 8.95f, 0.f));
	
	cb.finishBuild(curve);
	m_allGeo->setGeometry(curve, 0);
}

void FitTest::createRandomCurves()
{
    const unsigned n = 15 * 15;
	m_allGeo->create(n);
	
	BezierPatch bp;
	bp.resetCvs();
	
	int i=0;
	bp._contorlPoints[0].y += -.2f;
	bp._contorlPoints[1].y += -.4f;
	bp._contorlPoints[2].y += -.4f;
	bp._contorlPoints[3].y += -.5f;
	
	bp._contorlPoints[4].y += -.5f;
	bp._contorlPoints[5].y += .1f;
	bp._contorlPoints[6].y += .5f;
	bp._contorlPoints[7].y += .1f;
	
	bp._contorlPoints[9].y += .5f;
	bp._contorlPoints[10].y += .5f;
	
	bp._contorlPoints[13].y += -.4f;
	bp._contorlPoints[14].y += -.85f;
	bp._contorlPoints[15].y += -.21f;
	
	i=0;
	for(;i<16;i++) {
		bp._contorlPoints[i] *= 80.f;
		bp._contorlPoints[i].y += 10.f;
		bp._contorlPoints[i].z -= 10.f;
	}
	
	RandomCurve rc;
	rc.create(m_allGeo, 15, 15,
				&bp,
				Vector3F(-.15f, 4.f, 0.33f), 
				11, 17,
				.79f);
}

void FitTest::draw() 
{
    m_drawer->geometry(m_allGeo);
	// m_drawer->linearCurve(*m_curve);
	//m_drawer->smoothCurve(*m_curve, 8);
	//glBegin(GL_POINTS);
	//unsigned i=0;
	
	//glColor3f(0.f, 0.f, .5f);
	//for(;i<m_numSamples;i++)
	//	glVertex3fv((GLfloat *)&m_samples[i]);
	//glEnd();
	//glColor3f(0.8f, 0.f, 0.f);
	//for(i=1; i<m_numReducedP+1; i++)
		//m_drawer->arrow(m_reducedP[i-1], m_reducedP[i]);
	
	// int vv[2];
	// int ee[2];
	// float dV, dE;
	// Vector3F a, b, c, d;
	//for(i=0; i<m_numGroups; i++) {
		//drawOctahedron(m_octa[i]);
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
	//}
	
	glColor3f(.8f, .8f, .8f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	drawTetrahedron();
	glColor3f(.128f, .28f, .128f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
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

void FitTest::drawTetrahedron()
{
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
