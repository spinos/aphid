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

#include "BaseMesh.h"
#include "KdTreeDrawer.h"
#include <KdTree.h>
#include <Ray.h>
#include <SelectionArray.h>
#include <EasemodelUtil.h>
#include <AnchorGroup.h>
#include "FitDeformer.h"

static Vector3F rayo(15.299140, 20.149620, 97.618355), raye(-141.333694, -64.416885, -886.411499);
	
//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_tree = 0;
	m_mesh = new BaseMesh;
	m_drawer = new KdTreeDrawer;
	m_selected = new SelectionArray;
	m_selected->setComponentFilterType(PrimitiveFilter::TVertex);
	m_intersectCtx = new IntersectionContext;
	m_intersectCtx->setComponentFilterType(PrimitiveFilter::TVertex);
	m_anchors = new AnchorGroup;
	m_anchors->setHitTolerance(.8f);
	
	m_deformer = new FitDeformer;
	m_deformer->setAnchors(m_anchors);
	
#ifdef WIN32
	loadMesh("D:/aphid/mdl/flat.m");
#else
	loadMesh("/Users/jianzhang/aphid/mdl/flat.m");
#endif

	m_mode = SelectCompnent;
	
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(30);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientDraw()
{
	m_drawer->setCullFace(1);
	m_drawer->setWired(0);
	m_drawer->setGrey(0.4f);
	m_drawer->drawMesh(m_mesh);
	m_drawer->setWired(1);
	m_drawer->setGrey(0.9f);
	m_drawer->edge(m_mesh);
	m_drawer->setCullFace(0);
	
	m_drawer->setColor(0.f, 1.f, 0.4f);
	m_drawer->components(m_selected);
	if(m_anchors->numAnchors() > 0) {
		for(Anchor *a = m_anchors->firstAnchor(); m_anchors->hasAnchor(); a = m_anchors->nextAnchor())
			m_drawer->anchor(a, m_anchors->getHitTolerance());
	}
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
	rayo = origin;
	raye = origin + displacement;
	
	Ray ray(rayo, raye);
	if(m_mode == SelectCompnent) {
		if(!pickupComponent(ray, hit))
			m_selected->reset();
	}
	else {
		m_anchors->pickupAnchor(ray, hit);
	}
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
	if(m_mode == SelectCompnent) {
		Vector3F hit;
		pickupComponent(ray, hit);
	}
	else {
	    m_anchors->moveAnchor(stir);
		m_deformer->solve();
	}
}
//! [10]

void GLWidget::simulate()
{
    update();
}

void GLWidget::anchorSelected(float wei)
{
	if(m_selected->numVertices() < 1) return;
	Anchor *a = new Anchor(*m_selected);
	a->setWeight(wei);
	m_anchors->addAnchor(a);
	m_selected->reset();
}

void GLWidget::startDeform()
{
	if(m_anchors->numAnchors() < 1) {
		m_deformer->reset();
		rebuildTree();
		return;
	}
	
	if(m_targetAnchors) {
		std::vector<Anchor *> src; 
		for(Anchor *a = m_targetAnchors->firstAnchor(); m_targetAnchors->hasAnchor(); a = m_targetAnchors->nextAnchor())
			src.push_back(a);
		
		unsigned i = 0;
		for(Anchor *a = m_anchors->firstAnchor(); m_anchors->hasAnchor(); a = m_anchors->nextAnchor()) {
			if(i < src.size()) {
				fitAnchors(src[i], a);
			}
			i++;
		}	
	}
	
	m_deformer->precompute();	
	m_deformer->solve();
	
	rebuildTree();
}

bool GLWidget::pickupComponent(const Ray & ray, Vector3F & hit)
{
	m_intersectCtx->reset();
	if(!m_tree->intersect(ray, m_intersectCtx)) 
		return false;
	hit = m_intersectCtx->m_hitP;
	m_selected->add(m_intersectCtx->m_geometry, m_intersectCtx->m_componentIdx);
	return true;
}

void GLWidget::setTarget(AnchorGroup * src, KdTree * tree)
{
	m_targetAnchors = src;
	m_deformer->setTarget(tree);
}

void GLWidget::fit()
{
	m_deformer->fit();
	m_deformer->updateMesh();
}

void GLWidget::fitAnchors(Anchor * src, Anchor * dst)
{
	if(dst->numPoints() < 2 || src->numPoints() < 2) {
		dst->placeAt(src->getCenter());
		return;
	}
	
	BaseCurve dstCurve;
	for(unsigned i = 0; i < dst->numPoints(); i++)
		dstCurve.addVertex(dst->getPoint(i)->worldP);
		
	dstCurve.computeKnots();
	
	BaseCurve srcCurve;
	for(unsigned i = 0; i < src->numPoints(); i++)
		srcCurve.addVertex(src->getPoint(i)->worldP);
		
	srcCurve.computeKnots();
	
	dstCurve.fitInto(srcCurve);
	for(unsigned i = 0; i < dst->numPoints(); i++) {
		dst->getPoint(i)->worldP = dstCurve.getVertex(i);
	}
	
	dst->computeLocalSpace();
}

void GLWidget::removeLastAnchor()
{
	m_anchors->removeLast();
}

void GLWidget::rebuildTree()
{
	m_deformer->updateMesh();
	if(m_tree) delete m_tree;
	m_tree = new KdTree;
	m_tree->addMesh(m_mesh);
	m_tree->create();
}

void GLWidget::open()
{
	QFileDialog *fileDlg = new QFileDialog(this);
	QString temQStr = fileDlg->getOpenFileName(this, 
		tr("Open Model File"), "../", tr("Mesh(*.m)"));
	
	if(temQStr == NULL)
		return;
		
	loadMesh(temQStr.toStdString());
}

void GLWidget::loadMesh(std::string filename)
{
	EasyModel * eye = new EasyModel(filename.c_str());
	
	ESMUtil::copy(eye, m_mesh);
	
	delete eye;

	m_deformer->setMesh(m_mesh);
	
	m_selected->setTopology(m_deformer->getTopology());
	
	rebuildTree();
}
//:~
