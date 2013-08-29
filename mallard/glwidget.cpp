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
#include <MeshTopology.h>
#include "glwidget.h"
#include <AccPatchMesh.h>
#include <EasemodelUtil.h>
#include "zEXRImage.h"
#include <BezierDrawer.h>
#include <ToolContext.h>

//! [0]
GLWidget::GLWidget(QWidget *parent) : SingleModelView(parent)
{
	m_accmesh = new AccPatchMesh;
#ifdef WIN32
	std::string filename("D:/aphid/mdl/torus.m");
#else
	std::string filename("/Users/jianzhang/aphid/mdl/torus.m");
#endif

	loadMesh(filename);
	m_accmesh->setup(m_topo);
	
#ifdef WIN32
	_image = new ZEXRImage("D:/aphid/catmullclark/disp.exr");
#else
	_image = new ZEXRImage("/Users/jianzhang/aphid/catmullclark/disp.exr");
#endif
	//if(_image->isValid()) qDebug()<<"image is loaded";
	// use displacement map inside bezier drawer
	//_tess->setDisplacementMap(_image);

	m_fabricDrawer = new BezierDrawer;
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientDraw()
{
	getDrawer()->setGrey(1.f);
	getDrawer()->edge(mesh());
	
	m_fabricDrawer->drawAccPatchMesh(m_accmesh);
	drawSelection();
	//getDrawer()->drawKdTree(getTree());
}

void GLWidget::loadMesh(std::string filename)
{
	ESMUtil::ImportPatch(filename.c_str(), mesh());
	buildTopology();
	buildTree();
}

void GLWidget::setSelectionAsWale(int bothSide)
{
}

void GLWidget::changeWaleResolution(int change)
{
}

void GLWidget::changeCourseResolution(int change)
{
}

void GLWidget::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
    Vector3F rayo = origin;
	Vector3F raye = origin + displacement;
	
	Ray ray(rayo, raye);
	if(interactMode() == ToolContext::SelectVertex) {
		qDebug()<<"select vertex";
	}
	else {
		if(hitTest(ray, hit))
		    qDebug()<<"hit"<<hit.x<<" "<<hit.y<<" "<<hit.z;
		
	}
}

PatchMesh * GLWidget::mesh() const
{
	return m_accmesh;
}
//:~
