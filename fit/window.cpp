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
#include "TargetView.h"
#include "window.h"

//! [0]
Window::Window()
{
    glWidget = new GLWidget;
	targetWidget = new TargetView;
	
	QSplitter *splitter = new QSplitter;
	
	splitter->addWidget(glWidget);
    splitter->addWidget(targetWidget);

	setCentralWidget(splitter);
    setWindowTitle(tr("Matching Shape Version 0.6 Fri 5/3/2013"));
	
	glWidget->setTarget(targetWidget->getAnchors(), targetWidget->getTree());
	createActions();
	createMenus();
	
	//connect(m_control, SIGNAL(handleChanged(unsigned)), glWidget, SLOT(onHandleChanged(unsigned)));
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();
	
	QWidget::keyPressEvent(e);
}

void Window::createActions()
{
	loadTemplateAct = new QAction(tr("&Load Template"), this);
	loadTemplateAct->setStatusTip(tr("Open a model file as the template"));
	connect(loadTemplateAct, SIGNAL(triggered()), glWidget, SLOT(open()));
	
	loadTargetAct = new QAction(tr("&Load Target"), this);
	loadTargetAct->setStatusTip(tr("Open a model file as the target"));
	connect(loadTargetAct, SIGNAL(triggered()), targetWidget, SLOT(open()));
	
	saveTemplateAct = new QAction(tr("&Save Template"), this);
	saveTemplateAct->setStatusTip(tr("Save currest template as a model file"));
	connect(saveTemplateAct, SIGNAL(triggered()), glWidget, SLOT(save()));
}

void Window::createMenus()
{
	fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(loadTemplateAct);
	fileMenu->addAction(loadTargetAct);
	fileMenu->addAction(saveTemplateAct);
}

