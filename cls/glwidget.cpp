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

#include "KdTreeDrawer.h"
#include <RadialBasisFunction.h>
#include <Anchor.h>
#include <KdTree.h>
#include <IntersectionContext.h>

static Vector3F rayo(15.299140, 20.149620, 97.618355), raye(-141.333694, -64.416885, -886.411499);
	
//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(30);
	
	m_drawer = new KdTreeDrawer;
	
	rbf = new RadialBasisFunction;
	rbf->create(7);
	rbf->setXi(0, Vector3F(0,0,0));
	rbf->setXi(1, Vector3F(8,0,0));
	rbf->setXi(2, Vector3F(2.5,8.5,0));
	rbf->setXi(3, Vector3F(-8,0.2,1.0));
	rbf->setXi(4, Vector3F(-0.2,-8.2,-0.1));
	rbf->setXi(5, Vector3F(-0.1,-1.2, 7.1));
	rbf->setXi(6, Vector3F(9.0, 5.2, 2.1));
	
	rbf->setTau(16.0);
	rbf->computeWeights();
	
	m_anchor = new Anchor;
	Vector3F pa(-7.0, 5., 0.);
	m_anchor->placeAt(pa);
	
	rbf->solve(m_anchor->getCenter());
	
	m_mesh = new GeodesicSphereMesh(10);
	m_mesh->setRadius(8.f);
	
	m_tree = new KdTree;
	m_tree->addMesh(m_mesh);
	m_tree->create();
	
	m_intersectCtx = new IntersectionContext;
	m_intersectCtx->setComponentFilterType(PrimitiveFilter::TVertex);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientDraw()
{
	m_drawer->setGrey(0.6f);
	m_drawer->setWired(1);
	m_drawer->drawMesh(m_mesh);
	//m_drawer->drawKdTree(m_tree);
	unsigned nn = rbf->getNumNodes();
	for(unsigned i=0; i < nn; i++) {
		m_drawer->setGrey(0.8f);
		Vector3F p = rbf->getXi(i);
		m_drawer->solidCube(p.x, p.y, p.z, 1.f);
		float w = rbf->getResult(i);
		glColor3f(w,  1.f - w, 0.f);
		glBegin(GL_LINES);
		glVertex3f(p.x, p.y, p.z);
		glVertex3f(p.x, p.y + w * 8, p.z);
		glEnd();
	}
	m_drawer->setWired(0);
	m_drawer->setColor(0.f, 1.f, .4f);
	Vector3F ap = m_anchor->getCenter();
	m_drawer->solidCube(ap.x, ap.y, ap.z, 1.f);
	m_drawer->setColor(1.f, .1f, .1f);
	ap = m_intersectCtx->m_hitP;
	m_drawer->solidCube(ap.x, ap.y, ap.z, .2f);
	
	glBegin(GL_LINES);
	glVertex3f(ap.x, ap.y, ap.z);
	ap = m_anchor->getCenter();
	glVertex3f(ap.x, ap.y, ap.z);
	glVertex3f(ap.x, ap.y, ap.z);
	ap = m_intersectCtx->m_closest;
	glVertex3f(ap.x, ap.y, ap.z);
	glEnd();
	
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
	rayo = origin;
	raye = origin + displacement;
	
	Ray ray(rayo, raye);
	
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
	Ray ray(rayo, raye);
	float t;
	if(m_anchor->intersect(ray, t, 2.f)) {
		m_anchor->translate(stir);
		rbf->solve(m_anchor->getCenter());
		
		m_intersectCtx->reset();
		m_tree->closestPoint(m_anchor->getCenter(), m_intersectCtx);
	}
}
//! [10]

void GLWidget::simulate()
{
    update();
}

void GLWidget::anchorSelected(float wei)
{
}

void GLWidget::startDeform()
{
}

bool GLWidget::pickupAnchor(const Ray & ray, Vector3F & hit)
{
	return  false;
}

bool GLWidget::pickupComponent(const Ray & ray, Vector3F & hit)
{
	return false;
}

