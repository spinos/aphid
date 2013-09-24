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
#include "window.h"

//! [0]
Window::Window()
{
    glWidget = new GLWidget;
	m_tools = new ToolBox;
	
	GLWidget::InteractContext = m_tools;
	m_brushControl = new BrushControl(this); 
	
	addToolBar(m_tools);

	setCentralWidget(glWidget);
    setWindowTitle(tr("Mallard"));
    
	connect(m_tools, SIGNAL(contextChanged(int)), this, SLOT(receiveToolContext(int)));
    connect(m_tools, SIGNAL(actionTriggered(int)), this, SLOT(receiveToolAction(int)));
	connect(m_brushControl->pitchWidget(), SIGNAL(valueChanged(double)), glWidget, SLOT(receiveBrushPitch(double)));
	connect(m_brushControl->radiusWidget(), SIGNAL(valueChanged(double)), glWidget, SLOT(receiveBrushRadius(double)));
	connect(m_brushControl->numSamplesWidget(), SIGNAL(valueChanged(int)), glWidget, SLOT(receiveBrushNumSamples(int)));
	createActions();
	createMenus();
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
	else if(m_tools->previousContext() == ToolContext::CombBodyContourFeather || m_tools->previousContext() == ToolContext::ScaleBodyContourFeather)
		glWidget->deselectFeather();
}

void Window::receiveToolAction(int a)
{/*
    if(a == ToolContext::SetWaleEdge)
        glWidget->setSelectionAsWale(1);
    else if(a == ToolContext::SetSingleWaleEdge)
        glWidget->setSelectionAsWale(0);
	else if(a == ToolContext::IncreaseWale)
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
	showBrushControlAct = new QAction(tr("&Brush Control"), this);
	showBrushControlAct->setStatusTip(tr("Show brush settings"));
    connect(showBrushControlAct, SIGNAL(triggered()), m_brushControl, SLOT(show()));
}

void Window::createMenus()
{
	windowMenu = menuBar()->addMenu(tr("&Window"));
    windowMenu->addAction(showBrushControlAct);
}
