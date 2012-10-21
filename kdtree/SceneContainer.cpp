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
	Vector3F cubeC(-10.f, 22.f, -8.f);
	m_cube = new RandomMesh(171294, cubeC, 14.f, 0);
	
	Vector3F ballC(13.f, 3.f, 10.f);
	m_ball = new RandomMesh(211294, ballC, 13.f, 1);
	
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
