/*
 *  TargetView.cpp
 *  fit
 *
 *  Created by jian zhang on 5/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include <QtOpenGL>
#include "TargetView.h"
#include <MeshTopology.h>
	
//! [0]
TargetView::TargetView(QWidget *parent) : SingleModelView(parent)
{
	m_topo = new MeshTopology;
	QString filename  = QString("%1/mdl/ball.m")
                 .arg(QDir::currentPath());
	loadMesh(filename.toStdString().c_str());
}
//! [0]

//! [1]
TargetView::~TargetView() {}

void TargetView::buildTree()
{
	SingleModelView::buildTree();
	emit targetChanged();
}

void TargetView::loadMesh(std::string filename)
{
	SingleModelView::loadMesh(filename);
	
	m_topo->buildTopology(m_mesh);
	getSelection()->setTopology(m_topo->getTopology());
	m_topo->calculateNormal(m_mesh);
	buildTree();
}
//:~
