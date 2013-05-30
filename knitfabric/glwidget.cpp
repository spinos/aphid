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
#include "tessellator.h"
#include "KnitPatch.h"
#include "zEXRImage.h"
#include "FiberPatch.h"
#include <MeshTopology.h>

//! [0]
GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
#ifdef WIN32
_image = new ZEXRImage("D:/aphid/catmullclark/disp.exr");
#else
	_image = new ZEXRImage("/Users/jianzhang/aphid/catmullclark/disp.exr");
#endif
	if(_image->isValid()) qDebug()<<"image is loaded";
	
	_tess = new Tessellator();
	//_tess->setDisplacementMap(_image);
	
	_model = new PatchMesh;
	
#ifdef WIN32
	ESMUtil::ImportPatch("D:/aphid/mdl/plane.m", _model);
#else
	ESMUtil::ImportPatch("/Users/jianzhang/aphid/mdl/plane.m", _model);
#endif

    m_topo = new MeshTopology;
    m_topo->buildTopology(_model);
    m_topo->calculateNormal(_model);

	Vector3F* cvs = _model->getVertices();
	Vector3F* normal = _model->getNormals();
	float* ucoord = _model->us();
	float* vcoord = _model->vs();
	unsigned * uvIds = _model->uvIds();
	int numFace = _model->numPatches();

	AccStencil* sten = new AccStencil();
	AccPatch::stencil = sten;
	sten->setVertexPosition(cvs);
	sten->setVertexNormal(normal);
	
	sten->m_vertexAdjacency = m_topo->getTopology();

	_bezier = new AccPatch[numFace];
	unsigned * quadV = _model->quadIndices();
	for(int j = 0; j < numFace; j++) {
		sten->m_patchVertices[0] = quadV[0];
		sten->m_patchVertices[1] = quadV[1];
		sten->m_patchVertices[2] = quadV[2];
		sten->m_patchVertices[3] = quadV[3];
		
		_bezier[j].setTexcoord(ucoord, vcoord, &uvIds[j * 4]);
		_bezier[j].evaluateContolPoints();
		_bezier[j].evaluateTangents();
		_bezier[j].evaluateBinormals();
		_bezier[j].setCorner(_bezier[j].p(0, 0), 0);
		_bezier[j].setCorner(_bezier[j].p(3, 0), 1);
		_bezier[j].setCorner(_bezier[j].p(0, 3), 2);
		_bezier[j].setCorner(_bezier[j].p(3, 3), 3);
		
		quadV += 4;
	}
	
	m_knit = new KnitPatch;
	
	m_fiber = new FiberPatch[numFace];
	createFiber();
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::createFiber()
{
    const unsigned numFace = _model->numPatches();
    for(unsigned i = 0; i < numFace; i++) {
		_bezier[i].setUniformDetail(4.f);
		int seg = 8;
		_tess->setNumSeg(seg);
		_tess->evaluate(_bezier[i]);
	
		m_knit->setNumSeg(seg);
		
		Vector2F uvs[4];
        uvs[0] = _bezier[i].tex(0, 0);
        uvs[1] = _bezier[i].tex(1, 0);
        uvs[2] = _bezier[i].tex(1, 1);
        uvs[3] = _bezier[i].tex(0, 1);
        
        m_knit->directionByBiggestDu(uvs);
        m_knit->setThickness(0.24f);
        m_knit->createYarn(_tess->_positions, _tess->_normals);
        
        m_fiber[i].create(m_knit->getNumYarn(), m_knit->numPointsPerYarn());
		m_fiber[i].setThickness(0.16f);
        
        for(unsigned j = 0; j < m_knit->getNumYarn(); j++) {
            Vector3F *p = m_knit->yarnAt(j);
            Vector3F *n = m_knit->normalAt(j);
            Vector3F *t = m_knit->tangentAt(j);
            m_fiber[i].processYarn(j, p, n, t, m_knit->numPointsPerYarn());
        }
	}
}

void GLWidget::clientDraw()
{
	getDrawer()->edge(_model);
	
	drawBezier();
}

void GLWidget::drawBezier()
{
	float detail = 4.f;
	unsigned numFace = _model->numPatches();

	for(unsigned i = 0; i < numFace; i++) {
		//drawBezierPatchCage(_bezier[i]);
		_bezier[i].setUniformDetail(detail);
		drawBezierPatch(_bezier[i], detail);
		//drawYarn(_bezier[i], detail);
		//drawFiber(m_fiber[i]);
	}
}

void GLWidget::drawBezierPatch(AccPatch& patch, float detail)
{
	int maxLevel = (int)log2f(detail);
	//if(maxLevel + patch.getLODBase() > 10) {
	//	maxLevel = 10 - patch.getLODBase();
	//}
	int seg = 2;
	for(int i = 0; i < maxLevel; i++) {
		seg *= 2;
	}

	_tess->setNumSeg(seg);
	_tess->evaluate(patch);
	glColor3f(0.f, 0.3f, 0.9f);
	glEnable(GL_CULL_FACE);

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

void GLWidget::drawYarn(AccPatch& patch, float detail)
{
	int maxLevel = (int)log2f(detail);

	int seg = 2;
	for(int i = 0; i < maxLevel; i++) {
		seg *= 2;
	}

	_tess->setNumSeg(seg);
	_tess->evaluate(patch);
	
	m_knit->setNumSeg(seg);
	
	Vector2F uvs[4];
	uvs[0] = patch.tex(0, 0);
	uvs[1] = patch.tex(1, 0);
	uvs[2] = patch.tex(1, 1);
	uvs[3] = patch.tex(0, 1);
	
	m_knit->directionByBiggestDu(uvs);
	m_knit->setThickness(0.17f);
	m_knit->createYarn(_tess->_positions, _tess->_normals);
	
	
	for(unsigned i = 0; i < m_knit->getNumYarn(); i++) {
		Vector3F *p = m_knit->yarnAt(i);
		
		if(i%2 == 0) glColor3f(0.7f, 0.3f, 0.f);
		else glColor3f(0.4f, 0.7f, 0.3f);
	
		glEnableClientState( GL_VERTEX_ARRAY );
		glVertexPointer( 3, GL_FLOAT, 0, (float *)p);
		
		glDrawElements(GL_LINE_STRIP, m_knit->numPointsPerYarn(), GL_UNSIGNED_INT, m_knit->yarnIndices() );

		glDisableClientState( GL_VERTEX_ARRAY );
	}
}

void GLWidget::drawFiber(FiberPatch & fiber)
{
    glLineWidth(3.f);
    for(unsigned i = 0; i < fiber.getNumFiber(); i++) {
        unsigned ic = i / 6;
        if(ic % 3 == 0)glColor3f(0.f, 0.5f, 0.4f);
        else if(ic % 3 == 1)glColor3f(0.f, .4f, 0.6f);
        else glColor3f(0.f, 0.33f, 0.23f);
        
		Vector3F *p = fiber.fiberAt(i);
	
		glEnableClientState( GL_VERTEX_ARRAY );
		glVertexPointer( 3, GL_FLOAT, 0, (float *)p);
		
		glDrawElements(GL_LINE_STRIP, fiber.numPointsPerFiber(), GL_UNSIGNED_INT, fiber.fiberIndices() );

		glDisableClientState( GL_VERTEX_ARRAY );
	}
	glLineWidth(1.f);
}
