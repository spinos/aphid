/*
 *  SceneContainer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "SceneContainer.h"


SceneContainer::SceneContainer() 
{
	fDrawer = new ShapeDrawer;
	Vector3F cubeC(1.f, 22.f, -1.f);
	m_cube = new RandomMesh(255200, cubeC, 9.f, 1);
	
	Vector3F ballC(17.f, -5.f, 10.f);
	m_ball = new RandomMesh(261294, ballC, 8.f, 1);
	
	m_tree = new KdTree;
	m_tree->addMesh(m_cube);
	m_tree->addMesh(m_ball);
	m_tree->create();
}

SceneContainer::~SceneContainer() {}

void SceneContainer::renderWorld()
{
	fDrawer->box(32.f, 32.f, 32.f);
	fDrawer->setGrey(.3f);
	fDrawer->setWired(0);
	fDrawer->drawMesh(m_cube);
	fDrawer->drawMesh(m_ball);
	fDrawer->setWired(1);
	fDrawer->setColor(0.15f, 1.f, 0.5f);
	fDrawer->drawKdTree(m_tree);
}
