/*
 *  SceneEdit.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/13/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "SceneEdit.h"
#include "SceneTreeModel.h"
#include "MlScene.h"

SceneEdit::SceneEdit(MlScene * scene, QWidget *parent) : QDialog(parent) 
{
	setWindowTitle(tr("Scene Tree"));
	m_view = new QTreeView;
	m_model = new SceneTreeModel(scene);
	m_view->setModel(m_model);
	
	QHBoxLayout *layout = new QHBoxLayout;
	layout->addWidget(m_view);
	setLayout(layout);
	setContentsMargins(0, 0, 0, 0);
	layout->setContentsMargins(0, 0, 0, 0);
}

SceneEdit::~SceneEdit() {}