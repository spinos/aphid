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

#include "ControlWidget.h"
#include "KdTreeDrawer.h"
#include <KdTree.h>
#include <Ray.h>
#include <TargetGraph.h>
	
//! [0]
ControlWidget::ControlWidget(QWidget *parent) : Base3DView(parent)
{
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(update()));
	timer->start(30);

	m_drawer = new KdTreeDrawer;
	
	m_intersectCtx = new RayIntersectionContext;
	m_intersectCtx->setComponentFilterType(PrimitiveFilter::TFace);
	
	m_graph = new TargetGraph;
	m_graph->createVertices(3);
	m_graph->createIndices(3);
	m_graph->createTargetIndices(3);
	m_graph->createVertexWeights(3);
	m_graph->setVertex(0, 0, 0, 0);
	m_graph->setVertex(1, 4, 0, 0);
	m_graph->setVertex(2, 4, 4, 0); 
	m_graph->move(20, 20, 10);
	m_graph->setTriangle(0, 0, 1, 2);
	m_graph->setTargetTriangle(0, 0, 1, 1);
	m_graph->reset();
}
//! [0]

//! [1]
ControlWidget::~ControlWidget()
{
}

void ControlWidget::clientDraw()
{
	m_drawer->setGrey(0.5f);
	//m_drawer->drawKdTree(m_tree);
	//m_drawer->setWired(0);
	m_drawer->setColor(0.f, 1.f, 0.4f);
	
	glTranslatef(20,0,0);
	m_drawer->setWired(1);
	m_drawer->setColor(0.f, 1.f, .4f);
	
	glTranslatef(0,-20,0);
	m_drawer->setColor(0.f, 4.f, 1.f);
	
	
	m_drawer->setWired(1);
	m_drawer->setColor(0.f, 1.f, .3f);
	m_drawer->drawMesh(m_graph);
	
	Vector3F handp = m_graph->getHandlePos();
	m_drawer->setColor(.8f, 1.f, 0.f);
	m_drawer->solidCube(handp.x, handp.y, handp.z, 0.5f);
}
//! [7]

//! [9]
void ControlWidget::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
	Vector3F rayo = origin;
	Vector3F raye = origin + displacement;
	Ray ray(rayo, raye);
	if(!pickupControl(ray, hit)) return;
	updateControl();
}
//! [9]

void ControlWidget::clientDeselect()
{
}

//! [10]
void ControlWidget::clientMouseInput(Vector3F & origin, Vector3F & displacement, Vector3F & stir)
{
	Vector3F rayo = origin;
	Vector3F raye = origin + displacement;
	Ray ray(rayo, raye);
	Vector3F hit;
	if(!pickupControl(ray, hit)) return;
	updateControl();
}
//! [10]

bool ControlWidget::pickupControl(const Ray & ray, Vector3F & hit)
{
	m_intersectCtx->reset();
	if(m_graph->intersect(ray, m_intersectCtx)) {
	    hit = m_intersectCtx->m_hitP;
		return true;
	}
	return false;
}

void ControlWidget::updateControl()
{

}
