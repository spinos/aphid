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
#include "window.h"
#include <QDoubleEditSlider.h>
#include "SkeletonJointEdit.h"
#include "SkeletonPoseEdit.h"
//! [0]
Window::Window()
{
    glWidget = new GLWidget;
	alpha = new QDoubleEditSlider(tr("Rotate X"));
	alpha->setLimit(0.0, 360.0);
	alpha->setValue(0.0);
	beta = new QDoubleEditSlider(tr("Rotate Y"));
	beta->setLimit(0.0, 360.0);
	beta->setValue(0.0);
	gamma = new QDoubleEditSlider(tr("Rotate Z"));
	gamma->setLimit(0.0, 360.0);
	gamma->setValue(0.0);
	
	jointEdit = new SkeletonJointEdit(glWidget->skeleton());
	
	poseEdit = new SkeletonPoseEdit(glWidget->skeleton());
	
	QSplitter * page = new QSplitter;
	page->addWidget(glWidget);
	
	QVBoxLayout * layout = new QVBoxLayout;
	layout->addWidget(jointEdit);
	layout->addWidget(poseEdit);
	layout->setContentsMargins(0,0,0,0);
	
	QWidget * rgt = new QWidget;
	rgt->setLayout(layout);
	rgt->setContentsMargins(0,0,0,0);
	
	page->addWidget(rgt);
	
	setCentralWidget(page);
    setWindowTitle(tr("Skeleton Poses"));
    
    connect(glWidget, SIGNAL(jointSelected(int)), jointEdit, SLOT(attachToJoint(int)));
	connect(glWidget, SIGNAL(jointChanged()), jointEdit, SLOT(updateValues()));
	connect(jointEdit, SIGNAL(valueChanged()), glWidget, SLOT(updateJoint()));
	connect(poseEdit, SIGNAL(poseChanged()), glWidget, SLOT(update()));
	
	connect(alpha, SIGNAL(valueChanged(double)), glWidget, SLOT(setAngleAlpha(double)));
	connect(beta, SIGNAL(valueChanged(double)), glWidget, SLOT(setAngleBeta(double)));
	connect(gamma, SIGNAL(valueChanged(double)), glWidget, SLOT(setAngleGamma(double)));
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
        close();
	/*
	else if(e->key() == Qt::Key_A) {
		qDebug() << "anchor selected as one";
		glWidget->anchorSelected(1.f);
	}
	else if(e->key() == Qt::Key_Z) {
		qDebug() << "anchor selected as zero";
		glWidget->anchorSelected(0.f);
	}
	else if(e->key() == Qt::Key_D) {
		qDebug() << "calculate coordinate";
		glWidget->startDeform();
	}
    else if(e->key() == Qt::Key_R) {
		qDebug() << "rotate mode";
		glWidget->getSolver()->setInteractMode(DynamicsSolver::RotateJoint);
	}*/
	QWidget::keyPressEvent(e);
}
