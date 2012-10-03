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
#include <QtOpenGL>

#include <math.h>

#include "glwidget.h"

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

//! [0]
GLWidget::GLWidget(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    qtGreen = QColor::fromCmykF(0.40, 0.0, 1.0, 0.0);
    qtPurple = QColor::fromCmykF(0.29, 0.29, 0.20, 0.0);
	
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(40);
	
	fCamera = new BaseCamera();
	Vector3F eye(0.f, 0.f, 10.f);
	Vector3F coi(0.f, 0.f, 0.f);
	fCamera->lookFromTo(eye, coi);
	_aHemisphere = new HemisphereMesh(128, 256);
	_dome = new GeodesicHemisphereMesh;
	_vertexBuffer = new CUDABuffer;
	_program = new HemisphereProgram;
	_drawer = new ShapeDrawer;
	
	BRDFProgram::setVTheta(0.76f);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}
//! [1]

//! [2]
QSize GLWidget::minimumSizeHint() const
{
    return QSize(50, 50);
}
//! [2]

//! [3]
QSize GLWidget::sizeHint() const
//! [3] //! [4]
{
    return QSize(400, 400);
}
//! [4]

//! [6]
void GLWidget::initializeGL()
{
    qglClearColor(qtPurple.dark());

    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);
    //glEnable(GL_LIGHTING);
    //glEnable(GL_LIGHT0);
    glEnable(GL_MULTISAMPLE);
    //static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
    //glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
	
    CUDABuffer::setDevice();
    
	_vertexBuffer->create((float*)_aHemisphere->vertices(), _aHemisphere->getNumVertices() * 12);
}
//! [6]

//! [7]
void GLWidget::paintGL()
{
    _program->run(_vertexBuffer, _aHemisphere);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
	
	float m[16];
	fCamera->getMatrix(m);
	glMultMatrixf(m);
	
	glColor3f(0.f, 0.6f, 0.4f);
	_drawer->setWired(1);
	_drawer->drawMesh(_aHemisphere, _vertexBuffer);
	_drawer->setWired(0);
	
	glColor3f(0.f, 0.f, 1.f);
	glBegin(GL_LINES);
	glVertex3f(0.f, 0.f, 0.f);
	const Vector3F v = BRDFProgram::V;
	glVertex3f(v.x, v.y, v.z);
	glEnd();
	/*
	glColor3f(0.9f, 0.9f, 0.5f);
	glBegin(GL_QUADS);
	glVertex3f(-.5f, -.5f, 0.f);
	glVertex3f(.5f, -.5f, 0.f);
	glVertex3f(.5f, .5f, 0.f);
	glVertex3f(-.5f, .5f, 0.f);
	glEnd();*/
	
	_drawer->setWired(1);
	_drawer->drawMesh(_dome);
	_drawer->setWired(0);
	
	glFlush();
}
//! [7]

//! [8]
void GLWidget::resizeGL(int width, int height)
{
	glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	
	float aspect = (float)width/(float)height;
	float fov = 8.f;
	float right = fov/ 2.f;
	float top = right / aspect;

    glOrtho(-right, right, -top, top, 1.0, 1000.0);

    glMatrixMode(GL_MODELVIEW);
	
	fCamera->setPortWidth(width);
	fCamera->setPortHeight(height);
	fCamera->setHorizontalAperture(fov);
	fCamera->setVerticalAperture(fov/aspect);
}
//! [8]

//! [9]
void GLWidget::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}
//! [9]

//! [10]
void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
	const Qt::KeyboardModifiers modifiers = event->modifiers();
	if(modifiers == Qt::AltModifier) {
		moveCamera(event);
	}
	else {
		Vector3F injp(16, 16, 16);
		fCamera->intersection(event->x(), event->y(), injp);
		int dx = event->x() - lastPos.x();
		int dy = event->y() - lastPos.y();
		Vector3F injv;
		fCamera->screenToWorld(dx, dy, injv);
		//qDebug() << "screen hit:" << event->x() << " " << event->y();
		//qDebug() << "hit:" << injv.x << " " << injv.y << " " << injv.z;
	}

    lastPos = event->pos();
}
//! [10]

void GLWidget::moveCamera(QMouseEvent *event)
{
	int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();
	if (event->buttons() & Qt::LeftButton) {
        fCamera->tumble(dx, dy);
    } 
	else if (event->buttons() & Qt::MidButton) {
		fCamera->track(dx, dy);
    }
	else if (event->buttons() & Qt::RightButton) {
		fCamera->zoom(dy);
    }
}

void GLWidget::simulate()
{
    update();
}

void GLWidget::setProgram(HemisphereProgram * program)
{
    _program = program;
}