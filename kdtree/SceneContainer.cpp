/*
 *  SceneContainer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "SceneContainer.h"
#include <AllMath.h>
#include <RandomMesh.h>
#include <KdTree.h>
#include <KdTreeDrawer.h>
#include <BezierCurve.h>
#include <CurveBuilder.h>
#define TEST_CURVE 1
#define TEST_MESH 0
SceneContainer::SceneContainer(KdTreeDrawer * drawer) 
{
	m_drawer = drawer;
	m_tree = new KdTree;
	
#if TEST_MESH
	testMesh();
#endif

#if TEST_CURVE
	testCurve();
#endif
	
	m_tree->create();
}

SceneContainer::~SceneContainer() {}

void SceneContainer::testMesh()
{
	unsigned i=0;
	for(;i<4;i++) {
		Vector3F c(-10.f + 32.f * RandomF01(), 
					1.f + 32.f * RandomF01(), 
					-12.f + 32.f * RandomF01());
		m_mesh[i] = new RandomMesh(25000 + 5000 * RandomF01(), c, 4.f + 2.f * RandomF01(), i&1);
		m_tree->addGeometry(m_mesh[i]);
	}
}

void SceneContainer::testCurve()
{
	unsigned nv;
	Vector3F p, dp;
	CurveBuilder cb;
	unsigned i, j;
	for(i=0; i<399; i++) {
		m_curve[i] = new BezierCurve;
		nv = 10 + 5 * RandomF01();
		p.set(60.f + 60.f * RandomFn11(),
					14.f * RandomF01(),
					50.f + 60.f * RandomFn11());
		cb.addVertex(p);
		for(j=1; j< nv; j++) {
			dp.set(.32f * RandomFn11(),
					.3f + RandomF01() * 2.f,
					.92f * RandomFn11());
					
			p += dp;
			cb.addVertex(p);
		}
		cb.finishBuild(m_curve[i]);
		
		m_tree->addGeometry(m_curve[i]);
	}
}

void SceneContainer::renderWorld()
{
	m_drawer->setGrey(.3f);
	m_drawer->setWired(0);
	int i=0;
	
#if TEST_MESH
	for(;i<4;i++) 
		m_drawer->triangleMesh(m_mesh[i]);
#endif	
	glColor3f(0.1f, .2f, .3f);
	
#if TEST_CURVE
	for(i=0; i<399; i++)
		m_drawer->smoothCurve(*m_curve[i], 4);
#endif
		
	m_drawer->setWired(1);
	m_drawer->setColor(0.15f, 1.f, 0.5f);
	m_drawer->drawKdTree(m_tree);
}
