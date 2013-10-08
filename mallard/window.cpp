/****************************************************************************
**
** Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial Usage
** Licensees holding valid Qt Commercial licenses may use this file in
** accordance with the Qt Commercial License Agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and Nokia.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU Lesser General Public License version 2.1 requirements
** will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Nokia gives you certain additional
** rights.  These rights are described in the Nokia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met: http://www.gnu.org/copyleft/gpl.html.
**
** If you have questions regarding the use of this file, please contact
** Nokia at qt-info@nokia.com.
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtGui>

#include "glwidget.h"
#include "ToolBox.h"
#include "BrushControl.h"
#include "FeatherEdit.h"
#include "MlUVView.h"
#include "window.h"
#include "TimeControl.h"
//! [0]
Window::Window()
{
    glWidget = new GLWidget;
	m_tools = new ToolBox;
	
	glWidget->setInteractContext(m_tools);
	m_brushControl = new BrushControl(this);
	m_featherEdit = new FeatherEdit(this);
	MlUVView::FeatherLibrary = glWidget;
	m_timeControl = new TimeControl(this);
	glWidget->setPlayback(m_timeControl);
	addToolBar(m_tools);

	setCentralWidget(glWidget);
    setWorkTitle(tr("untitled"));
	createActions();
	createMenus();
    
	connect(m_tools, SIGNAL(contextChanged(int)), this, SLOT(receiveToolContext(int)));
    connect(m_tools, SIGNAL(actionTriggered(int)), this, SLOT(receiveToolAction(int)));
	connect(m_brushControl->pitchWidget(), SIGNAL(valueChanged(double)), glWidget, SLOT(receiveBrushPitch(double)));
	connect(m_brushControl->radiusWidget(), SIGNAL(valueChanged(double)), glWidget, SLOT(receiveBrushRadius(double)));
	connect(m_brushControl->numSamplesWidget(), SIGNAL(valueChanged(int)), glWidget, SLOT(receiveBrushNumSamples(int)));
	connect(glWidget, SIGNAL(sceneNameChanged(QString)), this, SLOT(setWorkTitle(QString)));
	connect(glWidget, SIGNAL(sendMessage(QString)), this, SLOT(showMessage(QString)));
	connect(m_featherEdit, SIGNAL(textureLoaded(QString)), glWidget, SLOT(receiveFeatherEditBackground(QString)));
	connect(glWidget, SIGNAL(sendFeatherEditBackground(QString)), m_featherEdit, SLOT(receiveTexture(QString)));
	connect(m_timeControl, SIGNAL(currentFrameChanged(int)), glWidget, SLOT(updateOnFrame(int)));
	
	statusBar()->showMessage(tr("Ready"));
}

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

void Window::receiveToolContext(int a)
{
	if(m_tools->previousContext() == ToolContext::EraseBodyContourFeather)
		glWidget->finishEraseFeather();
	glWidget->deselectFeather();
}

void Window::receiveToolAction(int a)
{
    if(a == ToolContext::RebuildBodyContourFeather)
        glWidget->rebuildFeather();
    else if(a == ToolContext::ClearBodyContourFeather)
        glWidget->clearFeather();
	/*else if(a == ToolContext::IncreaseWale)
		glWidget->changeWaleResolution(1);
	else if(a == ToolContext::DecreaseWale)
		glWidget->changeWaleResolution(-1);
	else if(a == ToolContext::IncreaseCourse)
		glWidget->changeCourseResolution(1);
	else if(a == ToolContext::DecreaseCourse)
		glWidget->changeCourseResolution(-1);
		*/
}

void Window::createActions()
{
    newSceneAct = new QAction(tr("&New"), this);
    newSceneAct->setStatusTip(tr("Create an empty scene"));
    connect(newSceneAct, SIGNAL(triggered()), glWidget, SLOT(cleanSheet()));
	
	showBrushControlAct = new QAction(tr("&Brush Control"), this);
	showBrushControlAct->setStatusTip(tr("Show brush settings"));
    connect(showBrushControlAct, SIGNAL(triggered()), m_brushControl, SLOT(show()));
	
	showFeatherEditAct = new QAction(tr("&Feather Editor"), this);
	showFeatherEditAct->setStatusTip(tr("Show feather editor"));
    connect(showFeatherEditAct, SIGNAL(triggered()), m_featherEdit, SLOT(show()));
	
	showTimeControlAct = new QAction(tr("&Time Control"), this);
	showTimeControlAct->setStatusTip(tr("Show time control"));
    connect(showTimeControlAct, SIGNAL(triggered()), m_timeControl, SLOT(show()));
	
	saveSceneAct = new QAction(tr("&Save"), this);
	saveSceneAct->setStatusTip(tr("Save current scene file"));
    connect(saveSceneAct, SIGNAL(triggered()), glWidget, SLOT(saveSheet()));
	
	saveSceneAsAct = new QAction(tr("&Save As"), this);
	saveSceneAsAct->setStatusTip(tr("Save current scene into another file"));
	connect(saveSceneAsAct, SIGNAL(triggered()), glWidget, SLOT(saveSheetAs()));
	
	importMeshAct = new QAction(tr("&Import Mesh"), this);
	importMeshAct->setStatusTip(tr("Load a mesh cache file as the body"));
	connect(importMeshAct, SIGNAL(triggered()), glWidget, SLOT(open()));
	
	openSceneAct = new QAction(tr("&Open"), this);
	openSceneAct->setStatusTip(tr("Load a file as current scene"));
	connect(openSceneAct, SIGNAL(triggered()), glWidget, SLOT(openSheet()));
	
	revertAct = new QAction(tr("&Revert To Saved"), this);
	revertAct->setStatusTip(tr("Discard changes to current scene after lastest save"));
    connect(revertAct, SIGNAL(triggered()), glWidget, SLOT(revertSheet()));
	
	importBakeAct = new QAction(tr("&Import Baked Animation"), this);
	importBakeAct->setStatusTip(tr("Attach bake point cache to the body"));
    connect(importBakeAct, SIGNAL(triggered()), glWidget, SLOT(chooseBake()));
}

void Window::createMenus()
{
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(newSceneAct);
	fileMenu->addAction(openSceneAct);
	fileMenu->addAction(saveSceneAct);
	fileMenu->addAction(saveSceneAsAct);
	fileMenu->addAction(revertAct);
	fileMenu->addSeparator();
	fileMenu->addAction(importMeshAct);
	fileMenu->addAction(importBakeAct);
	
	windowMenu = menuBar()->addMenu(tr("&Window"));
    windowMenu->addAction(showFeatherEditAct);
	windowMenu->addAction(showBrushControlAct);
	windowMenu->addAction(showTimeControlAct);
}

void Window::setWorkTitle(QString name)
{
	setWindowTitle(QString("Mallard - %1").arg(name));
}

void Window::showMessage(QString msg)
{
	statusBar()->showMessage(msg);
}
