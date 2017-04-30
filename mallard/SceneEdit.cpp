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
#include "SceneTreeParser.h"
#include "MlScene.h"
#include "EditDelegate.h"

SceneEdit::SceneEdit(MlScene * scene, QWidget *parent) : QDialog(parent) 
{
	setWindowTitle(tr("Scene Tree"));
	m_view = new QTreeView;
	EditDelegate * delegate = new EditDelegate;
	
	QStringList headers;
	headers<<"attribute"<<"value";
	m_model = new SceneTreeParser(headers, scene);
	m_view->setModel(m_model);
	
	m_view->setItemDelegate(delegate);
	
	QHBoxLayout *layout = new QHBoxLayout;
	layout->addWidget(m_view);
	setLayout(layout);
	setContentsMargins(0, 0, 0, 0);
	layout->setContentsMargins(0, 0, 0, 0);
	
	connect(delegate, SIGNAL(commitData(QWidget *)),
            m_model, SLOT(receiveData(QWidget *)));
}

SceneEdit::~SceneEdit() {}

void SceneEdit::reloadScene()
{
	m_model->rebuild();
}

QObject * SceneEdit::model() const
{
	return m_model;
}
