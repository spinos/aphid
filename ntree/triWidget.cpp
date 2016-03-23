/*
 *  TriWidget.cpp
 *  
 *
 *  Created by jian zhang on 3/23/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include "triWidget.h"
#include <BaseCamera.h>
#include <GeoDrawer.h>
#include <NTreeDrawer.h>

TriWidget::TriWidget(const std::string & filename, QWidget *parent) : 
Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
    m_intersectCtx.m_success = 0;
	
	if(filename.size() > 1) m_container.readTree(filename);
}

TriWidget::~TriWidget()
{}

void TriWidget::clientInit()
{ connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update())); }

void TriWidget::clientDraw()
{
	//drawBoxes();
	drawTree();
	// drawIntersect();
	// drawGrid();
}

void TriWidget::drawTree()
{
	if(!m_container.tree() ) return; 
	
	getDrawer()->setColor(.15f, .25f, .35f);
	getDrawer()->boundingBox(m_container.tree()->getBBox() );
	
	NTreeDrawer dr;
	dr.drawTree<cvx::Triangle>(m_container.tree() );
}
