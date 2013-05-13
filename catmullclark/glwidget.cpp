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
#include <BaseMesh.h>
#include <EasemodelUtil.h>
#include "subdivision.h"
#include "accPatch.h"
#include "accStencil.h"
#include "patchTopology.h"
#include "LODCamera.h"
#include "tessellator.h"
#include "zEXRImage.h"
#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

//! [0]
GLWidget::GLWidget(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    xRot = 0;
    yRot = 0;
    zRot = 0;

    qtGreen = QColor::fromCmykF(0.40, 0.0, 1.0, 0.0);
    qtPurple = QColor::fromCmykF(0.29, 0.29, 0.20, 0.0);

	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(40);
	
	_image = new ZEXRImage("/Users/jianzhang/aphid/catmullclark/disp.exr");
	if(_image->isValid()) qDebug()<<"image is loaded";
	
	_camera = new LODCamera();
	_camera->setClip(1.f, 1000.f);
	_camera->translate(0.f, 0.f, 100.f);
	_tess = new Tessellator();
	_tess->setDisplacementMap(_image);
	
	//_subdiv = new Subdivision();
	//_subdiv->setLevel(4);
	//_subdiv->runTest();
	_model = new BaseMesh;
	ESMUtil::Import("/Users/jianzhang/aphid/catmullclark/plane.m", _model);

	Vector3F* cvs = _model->getVertices();
	Vector3F* normal = _model->getNormals();
	int* valence = _model->getVertexValence();
	int* patchV = _model->getPatchVertex();
	char* patchB = _model->getPatchBoundary();
	float* ucoord = _model->getUs();
	float* vcoord = _model->getVs();
	int* uvIds = _model->getUVIds();
	int pv[24];
	char pb[15];
	float pp[24 * 3];
	int numFace = _model->getNumFaces();
	//numFace = 1;
	//_mesh = new Subdivision[numFace];
	_topo = new PatchTopology[numFace];
	AccStencil* sten = new AccStencil();
	AccPatch::stencil = sten;
	sten->setVertexPosition(cvs);
	sten->setVertexNormal(normal);
	
	for(int j = 0; j < numFace; j++)
	{
		//_model->setPatchAtFace(j, pv, pb);
		//for(int i = 0; i < 24; i++)
		{
		//	pp[i * 3] = cvs[pv[i] * 3];
		//	pp[i * 3 + 1] = cvs[pv[i] * 3 + 1];
		//	pp[i * 3 + 2] = cvs[pv[i] * 3 + 2];
		}
		//_mesh[j].setLevel(1);
		//_mesh[j].setPatch(pp, pv, pb, valence);
		//_mesh[j].dice();
		
		_topo[j].setVertexValence(valence);
		
		int* ip = patchV;
		ip += j * 24;
		_topo[j].setVertex(ip);
		
		char* cp = patchB;
		cp += j * 15;
		_topo[j].setBoundary(cp);
	}
	/*
	_mesh1 = new Subdivision[numFace];
	for(int j = 0; j < numFace; j++)
	{
		_model->setPatchAtFace(j, pv, pb);
		for(int i = 0; i < 24; i++)
		{
			pp[i * 3] = cvs[pv[i] * 3];
			pp[i * 3 + 1] = cvs[pv[i] * 3 + 1];
			pp[i * 3 + 2] = cvs[pv[i] * 3 + 2];
		}
		_mesh1[j].setLevel(4);
		_mesh1[j].setPatch(pp, pv, pb, valence);
		_mesh1[j].dice();
	}*/
	_bezier = new AccPatch[numFace];
	for(int j = 0; j < numFace; j++)
	{
		_bezier[j].setTexcoord(ucoord, vcoord, &uvIds[j * 4]);
		_bezier[j].evaluateContolPoints(_topo[j]);
		_bezier[j].evaluateTangents();
		_bezier[j].evaluateBinormals();
		_bezier[j].setCorner(_bezier[j].p(0, 0), 0);
		_bezier[j].setCorner(_bezier[j].p(3, 0), 1);
		_bezier[j].setCorner(_bezier[j].p(0, 3), 2);
		_bezier[j].setCorner(_bezier[j].p(3, 3), 3);
	}
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

//! [5]
void GLWidget::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != xRot) {
        xRot = angle;
        emit xRotationChanged(angle);
        updateGL();
    }
}
//! [5]

void GLWidget::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != yRot) {
        yRot = angle;
        emit yRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != zRot) {
        zRot = angle;
        emit zRotationChanged(angle);
        updateGL();
    }
}

//! [6]
void GLWidget::initializeGL()
{
    qglClearColor(qtPurple.dark());

    

    glEnable(GL_DEPTH_TEST);
    
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
	
	glLoadMatrixf(_camera->getMatrix());


    //glTranslatef(0.0, 0.0, -100.0);
    //glRotatef(xRot / 16.0, 1.0, 0.0, 0.0);
    //glRotatef(yRot / 16.0, 0.0, 1.0, 0.0);
    //glRotatef(zRot / 16.0, 0.0, 0.0, 1.0);
	
	
	// draw origin coordinate system
	glBegin(GL_LINES);
	glColor3f(1.,0.,0.);
	glVertex3i(0,0,0);
	glVertex3i(100,0,0);
	glColor3f(0.,1.,0.);
	glVertex3i(0,0,0);
	glVertex3i(0,100,0);
	glColor3f(0.,0.,1.);
	glVertex3i(0,0,0);
	glVertex3i(0,0,100);
	glEnd();
	
	//_subdiv->draw();
	drawModel();
	
	//drawMesh();
	drawBezier();

	glFlush();
}
//! [7]

//! [8]
void GLWidget::resizeGL(int width, int height)
{
    //int side = qMin(width, height);
    //glViewport((width - side) / 2, (height - side) / 2, side, side);
	glViewport(0, 0, width, height);
	_camera->setViewport(55.f, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
/*	
	float aspect = (float)width/(float)height;
	float fov = 10.f;
	float right = fov/ 2.f;
	float top = right / aspect;
#ifdef QT_OPENGL_ES_1
    glOrthof(-0.5, +0.5, -0.5, +0.5, 4.0, 15.0);
#else
    glOrtho(-right, right, -top, top, 1.0, 1000.0);
#endif
*/
	const float* frustum = _camera->getFrustum();
	glFrustum(frustum[0], frustum[1], frustum[2], frustum[3], frustum[4], frustum[5]);
    glMatrixMode(GL_MODELVIEW);
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
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        _camera->tumble(dx, dy);
    } else if (event->buttons() & Qt::RightButton) {
        _camera->dolly(dy);
    }
	else if (event->buttons() & Qt::MidButton) {
        _camera->track(dx, dy);
    }
    lastPos = event->pos();
	update();
}
//! [10]
void GLWidget::simulate()
{
    //update();
    
}

void GLWidget::drawModel()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	unsigned numFace = _model->getNumFaces();
	int* counts = _model->getFaceCount();
	int* connection = _model->getFaceConnection();
	float* cvs = _model->getVertexPosition();
	glColor3f(.1f, 1.f, .8f);
	glBegin(GL_QUADS);
	int acc = 0;
	for(unsigned i=0; i < numFace; i++)
	{
		for(int j = 0; j < counts[i]; j++)
		{
			int vert = connection[acc];
			glVertex3f(cvs[vert * 3], cvs[vert * 3 + 1], cvs[vert * 3 + 2]);
			acc++;
		}
	}
	glEnd();
	
}

void GLWidget::drawMesh()
{
	unsigned numFace = _model->getNumFaces();
	for(unsigned i = 0; i < numFace; i++)
	{
		_mesh1[i].draw();
	}
}

void GLWidget::drawBezier()
{
	Vector3F bmin, bmax;
	unsigned numFace = _model->getNumFace();
	for(unsigned i = 0; i < numFace; i++)
	{
		//drawBezierPatchCage(_bezier[i]);
		_camera->computeQuadLOD(_bezier[i]);
		float detail = _bezier[i].getMaxLOD();
		if(detail > 0) drawBezierPatch(_bezier[i], detail);
	}
}

void normalColor(Vector3F& nor)
{
	glColor3f(nor.x , nor.y , nor.z);
}

void GLWidget::drawBezierPatch(AccPatch& patch, float detail)
{
	int maxLevel = (int)log2f(detail) + 1;
	if(maxLevel + patch.getLODBase() > 10) {
		maxLevel = 10 - patch.getLODBase();
	}
	int seg = 2;
	for(int i = 0; i < maxLevel; i++)
	{
		seg *= 2;
	}
	_tess->setNumSeg(seg);
	_tess->evaluate(patch);
	glEnable(GL_CULL_FACE);
	glColor3f(0.5f, 0.1f, 0.2f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 0, _tess->getPositions() );
	
	glEnableClientState( GL_COLOR_ARRAY );
	glColorPointer( 3, GL_FLOAT, 0, _tess->getNormals() );

	glDrawElements( GL_QUADS, seg * seg * 4, GL_UNSIGNED_INT, _tess->getVertices() );
	glDisableClientState( GL_COLOR_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );
}

void GLWidget::drawBezierPatchCage(AccPatch& patch)
{
	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_QUADS);
	glColor3f(1,1,1);
	for(unsigned j=0; j < 3; j++)
	{
		for(unsigned i = 0; i < 3; i++)
		{
			Vector3F p = patch.p(i, j);
			glVertex3f(p.x, p.y, p.z);
			p = patch.p(i + 1, j);
			glVertex3f(p.x, p.y, p.z);
			p = patch.p(i + 1, j + 1);
			glVertex3f(p.x, p.y, p.z);
			p = patch.p(i, j + 1);
			glVertex3f(p.x, p.y, p.z);
		}
	}
	glEnd();
}