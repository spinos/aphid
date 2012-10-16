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
	m_mesh = new RandomMesh(64);
	m_tree = new KdTree;
	m_tree->create(m_mesh);
}

SceneContainer::~SceneContainer() {}

void SceneContainer::renderWorld()
{
	fDrawer->box(32.f, 32.f, 32.f);
	fDrawer->setGrey(1.f);
	fDrawer->drawMesh(m_mesh);
	fDrawer->drawKdTree(m_tree);
}
