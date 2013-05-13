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
#include <PatchMesh.h>
#include <EasemodelUtil.h>
#include "accPatch.h"
#include "accStencil.h"
#include "patchTopology.h"
#include "tessellator.h"
#include "zEXRImage.h"
#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

//! [0]
GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{	
	_image = new ZEXRImage("/Users/jianzhang/aphid/catmullclark/disp.exr");
	if(_image->isValid()) qDebug()<<"image is loaded";
	
	_tess = new Tessellator();
	_tess->setDisplacementMap(_image);
	
	_model = new PatchMesh;
	ESMUtil::ImportPatch("/Users/jianzhang/aphid/catmullclark/plane.m", _model);

	Vector3F* cvs = _model->getVertices();
	Vector3F* normal = _model->getNormals();
	
	unsigned* valence = _model->vertexValence();
	unsigned* patchV = _model->patchVertices();
	char* patchB = _model->patchBoundaries();
	float* ucoord = _model->us();
	float* vcoord = _model->vs();
	unsigned * uvIds = _model->uvIds();
	int numFace = _model->numPatches();
	//numFace = 1;
	//_mesh = new Subdivision[numFace];
	_topo = new PatchTopology[numFace];
	AccStencil* sten = new AccStencil();
	AccPatch::stencil = sten;
	sten->setVertexPosition(cvs);
	sten->setVertexNormal(normal);
	
	for(int j = 0; j < numFace; j++) {
		_topo[j].setVertexValence(valence);
		
		unsigned* ip = patchV;
		ip += j * 24;
		_topo[j].setVertex(ip);
		
		char* cp = patchB;
		cp += j * 15;
		_topo[j].setBoundary(cp);
	}

	_bezier = new AccPatch[numFace];
	for(int j = 0; j < numFace; j++) {
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

void GLWidget::clientDraw()
{
	getDrawer()->edge(_model);
	
	drawBezier();
}

void GLWidget::drawBezier()
{
	float detail = 7.f;
	unsigned numFace = _model->numPatches();
	for(unsigned i = 0; i < numFace; i++)
	{
		_bezier[i].setUniformDetail(detail);
		drawBezierPatch(_bezier[i], detail);
	}
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
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
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