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

#include "MeshLaplacian.h"
#include "LaplaceDeformer.h"
#include "KdTreeDrawer.h"
#include <KdTree.h>
#include <Ray.h>
#include <RayIntersectionContext.h>
#include <SelectionArray.h>

static Vector3F rayo(15.299140, 20.149620, 97.618355), raye(-141.333694, -64.416885, -886.411499);
 

static RayIntersectionContext intersectCtx;
	
//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(30);
	
#ifdef WIN32
	m_mesh = new MeshLaplacian("D:/aphid/lapl/cube.m");
#else	
	m_mesh = new MeshLaplacian("/Users/jianzhang/aphid/lapl/cube.m");
#endif
	m_drawer = new KdTreeDrawer;
	m_deformer = new LaplaceDeformer;
	
	m_deformer->setMesh(m_mesh);
	m_deformer->solve();
	
	m_tree = new KdTree;
	m_tree->addMesh(m_mesh);
	m_tree->create();
	
	//Ray ray(rayo, raye);
	//m_tree->intersect(ray, intersectCtx);
	
	m_selected = new SelectionArray();
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}
//! [1]

//! [7]
void GLWidget::clientDraw()
{
    m_drawer->setWired(1);
	m_drawer->setGrey(0.9f);
    m_drawer->drawMesh(m_mesh);
	m_drawer->drawMesh(m_mesh, m_deformer);
	//m_drawer->setGrey(0.5f);
	//m_drawer->drawKdTree(m_tree);
	
	/*glBegin(GL_LINES);
	glColor3f(1,0,0);
	glVertex3f(rayo.x, rayo.y, rayo.z);
	glColor3f(0,0,1);
	glVertex3f(raye.x, raye.y, raye.z);
	glEnd();*/
	m_drawer->setWired(0);
	m_drawer->setColor(0.f, 1.f, 0.4f);

	m_drawer->components(m_selected);
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
	rayo = origin;
	raye = origin + displacement;
	
	Ray rayy(rayo, raye);
	intersectCtx.reset();
	if(m_tree->intersect(rayy, intersectCtx)) {
		m_selected->add(intersectCtx.m_primitive);
		hit = intersectCtx.m_hitP;
	}
	else
		m_selected->reset();
}
//! [9]

void GLWidget::clientDeselect()
{

}

//! [10]
void GLWidget::clientMouseInput(Vector3F & origin, Vector3F & displacement, Vector3F & stir)
{
	rayo = origin;
	raye = origin + displacement;
	Ray rayy(rayo, raye);
	intersectCtx.reset();
	if(m_tree->intersect(rayy, intersectCtx)) {
		m_selected->add(intersectCtx.m_primitive);
	}
}
//! [10]

void GLWidget::simulate()
{
    update();
}


