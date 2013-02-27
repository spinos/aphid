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
	
	_dynamics = new DynamicsSolver();
	_dynamics->initPhysics();
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(40);
	fCamera = new BaseCamera();
	
	
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

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

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
	
	
}
//! [6]

//! [7]
void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float m[16];
	fCamera->getMatrix(m);
	glMultMatrixf(m);
	_dynamics->renderWorld();
	glFlush();
}
//! [7]

//! [8]
void GLWidget::resizeGL(int width, int height)
{
    //int side = qMin(width, height);
    //glViewport((width - side) / 2, (height - side) / 2, side, side);
	glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	
	float aspect = (float)width/(float)height;
	float fov = 40.f;
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
    if(event->modifiers() == Qt::AltModifier) 
        return;
    
    processSelection(event);
    if(_dynamics->getInteractMode() == DynamicsSolver::ToggleLock) 
        processLock(event);
}
//! [9]

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    _dynamics->removeSelection();
    if(_dynamics->getInteractMode() == DynamicsSolver::RotateJoint) {
        _dynamics->removeTorque();
    }
}

//! [10]
void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if(event->modifiers() == Qt::AltModifier) {
        processCamera(event);
    }
    else if(_dynamics->getInteractMode() == DynamicsSolver::RotateJoint) {
        processTorque(event);
    }
    else 
        processImpulse(event);

    lastPos = event->pos();
}
//! [10]

void GLWidget::processCamera(QMouseEvent *event)
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

void GLWidget::processSelection(QMouseEvent *event)
{
    Vector3F origin, incident;
    fCamera->incidentRay(event->x(), event->y(), origin, incident);
    incident = incident.normal() * 1000.f;
    _dynamics->selectByRayHit(origin, incident, m_hitPosition);
}

void GLWidget::processImpulse(QMouseEvent *event)
{
    if(!_dynamics->hasActive())
        return;
    
    fCamera->intersection(event->x(), event->y(), m_hitPosition);
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();
    Vector3F injv;
    fCamera->screenToWorld(dx, dy, injv);
    injv *= 10.f;
    _dynamics->addImpulse(injv);
    //qDebug() << "force:" << injv.x << " " << injv.y << " " << injv.z;
}

void GLWidget::processTorque(QMouseEvent *event)
{
    if(!_dynamics->hasActive())
        return;
    
    fCamera->intersection(event->x(), event->y(), m_hitPosition);
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();
    Vector3F injv;
    fCamera->screenToWorld(dx, dy, injv);
    _dynamics->addTorque(injv);
    //qDebug() << "force:" << injv.x << " " << injv.y << " " << injv.z;
}

void GLWidget::processLock(QMouseEvent *event)
{
    _dynamics->toggleMassProp();
}

void GLWidget::simulate()
{
    update();
    _dynamics->simulate();
}

DynamicsSolver* GLWidget::getSolver()
{
    return _dynamics;
}
