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

#include "TargetView.h"

#include "BaseMesh.h"
#include "KdTreeDrawer.h"
#include <KdTree.h>
#include <Ray.h>
#include <EasemodelUtil.h>
#include <AnchorGroup.h>
#include <MeshTopology.h>

static Vector3F rayo(15.299140, 20.149620, 97.618355), raye(-141.333694, -64.416885, -886.411499);
	
//! [0]
TargetView::TargetView(QWidget *parent) : Base3DView(parent)
{
	m_tree = 0;
	m_mesh = new BaseMesh;
	m_anchors = new AnchorGroup;
	m_anchors->setHitTolerance(.8f);
	m_topo = new MeshTopology;
	
	QString filename  = QString("%1/mdl/ball.m")
                 .arg(QDir::currentPath());

	loadMesh(filename.toStdString().c_str());
	
	m_mode = SelectCompnent;
}
//! [0]

//! [1]
TargetView::~TargetView()
{
}

void TargetView::clientDraw()
{
	KdTreeDrawer *drawer = getDrawer();
	drawer->setCullFace(1);
	drawer->setWired(0);
	drawer->setGrey(0.4f);
	drawer->drawMesh(m_mesh);
	drawer->setWired(1);
	drawer->setGrey(0.9f);
	drawer->edge(m_mesh);
	drawer->setCullFace(0);
	
	drawSelection();
	drawAnchors();
}
//! [7]

//! [9]
void TargetView::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
	rayo = origin;
	raye = origin + displacement;
	
	Ray ray(rayo, raye);
	if(m_mode == SelectCompnent) {
		pickupComponent(ray, hit);
	}
	else {
		m_anchors->pickupAnchor(ray, hit);
	}
}
//! [9]

void TargetView::clientDeselect()
{

}

//! [10]
void TargetView::clientMouseInput(Vector3F & origin, Vector3F & displacement, Vector3F & stir)
{
	rayo = origin;
	raye = origin + displacement;
	Ray ray(rayo, raye);
	if(m_mode == SelectCompnent) {
		Vector3F hit;
		pickupComponent(ray, hit);
	}
}

void TargetView::sceneCenter(Vector3F & dst) const
{
    dst.x = m_tree->m_bbox.getMin(0) * 0.5f + m_tree->m_bbox.getMax(0) * 0.5f;
    dst.y = m_tree->m_bbox.getMin(1) * 0.5f + m_tree->m_bbox.getMax(1) * 0.5f;
    dst.z = m_tree->m_bbox.getMin(2) * 0.5f + m_tree->m_bbox.getMax(2) * 0.5f;
}

void TargetView::simulate()
{
    update();
}

void TargetView::anchorSelected(float wei)
{
	if(getSelection()->numVertices() < 1) return;
	Anchor *a = new Anchor(*getSelection());
	a->setWeight(wei);
	m_anchors->addAnchor(a);
	clearSelection();
}

bool TargetView::pickupComponent(const Ray & ray, Vector3F & hit)
{
	getIntersectionContext()->reset();
	if(!m_tree->intersect(ray, getIntersectionContext())) 
		return false;
		
	hit = getIntersectionContext()->m_hitP;
	addHitToSelection();
	return true;
}

AnchorGroup * TargetView::getAnchors() const
{
	return m_anchors;
}

KdTree * TargetView::getTree() const
{
	return m_tree;
}

void TargetView::buildTree()
{
	if(m_tree) delete m_tree;
	m_tree = new KdTree;
	m_tree->addMesh(m_mesh);
	m_tree->create();
	emit targetChanged();
}

void TargetView::open()
{
	QFileDialog *fileDlg = new QFileDialog(this);
	QString temQStr = fileDlg->getOpenFileName(this, 
		tr("Open Model File"), "../", tr("Mesh(*.m)"));
	
	if(temQStr == NULL)
		return;
		
	loadMesh(temQStr.toStdString());
}

void TargetView::loadMesh(std::string filename)
{
	ESMUtil::Import(filename.c_str(), m_mesh);
	
	m_topo->buildTopology(m_mesh);
	getSelection()->setTopology(m_topo->getTopology());
	m_topo->calculateNormal(m_mesh);
	buildTree();
}

void TargetView::keyPressEvent(QKeyEvent *e)
{
	if(e->key() == Qt::Key_A) {
		anchorSelected(1.f);
	}
	else if(e->key() == Qt::Key_Z) {
		m_anchors->removeLast();
	}
	else if(e->key() == Qt::Key_X) {
		m_anchors->removeActive();
	}

	Base3DView::keyPressEvent(e);
}

void TargetView::setSelectComponent()
{
	m_mode = SelectCompnent;
	m_anchors->clearSelected();
}

void TargetView::setSelectAnchor()
{
	m_mode = TransformAnchor;
}

void TargetView::drawAnchors()
{
	if(m_anchors->numAnchors() < 1) return;
	
	Anchor *a;
	for(a = m_anchors->firstAnchor(); m_anchors->hasAnchor(); a = m_anchors->nextAnchor())
		getDrawer()->anchor(a, a == m_anchors->activeAnchor());

	if(m_mode == SelectCompnent) return;
	
	getDrawer()->spaceHandle(m_anchors->activeAnchor());
}
//:~
