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
#include <KdTree.h>
#include <Ray.h>
#include <SelectionArray.h>
#include <EasemodelUtil.h>
#include <AnchorGroup.h>
#include "FitDeformer.h"
	
//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_tree = 0;
	m_mesh = new BaseMesh;
	
	m_anchors = new AnchorGroup;
	m_anchors->setHitTolerance(.8f);
	
	m_deformer = new FitDeformer;
	m_deformer->setAnchors(m_anchors);
	
#ifdef WIN32
	loadMesh("D:/aphid/mdl/face.m");
#else
	loadMesh("/Users/jianzhang/aphid/mdl/face.m");
#endif

	m_mode = SelectCompnent;
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientDraw()
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
	
	if(m_anchors->numAnchors() > 0) {
		for(Anchor *a = m_anchors->firstAnchor(); m_anchors->hasAnchor(); a = m_anchors->nextAnchor())
			drawer->anchor(a, m_anchors->getHitTolerance());
	}
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
	Vector3F rayo = origin;
	Vector3F raye = origin + displacement;
	
	Ray ray(rayo, raye);
	if(m_mode == SelectCompnent) {
		pickupComponent(ray, hit);
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
	Vector3F rayo = origin;
	Vector3F raye = origin + displacement;
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
	if(getSelection()->numVertices() < 1) return;
	Anchor *a = new Anchor(*getSelection());
	a->setWeight(wei);
	m_anchors->addAnchor(a);
	clearSelection();
}

void GLWidget::startDeform()
{
	if(m_anchors->numAnchors() < 1) {
		m_deformer->reset();
		buildTree();
		return;
	}
	
	if(m_targetAnchors->numAnchors() < 1) return;
	
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
	buildTree();
	m_deformer->calculateNormal(m_mesh);
}

bool GLWidget::pickupComponent(const Ray & ray, Vector3F & hit)
{
	getIntersectionContext()->reset();
	if(!m_tree->intersect(ray, getIntersectionContext())) 
		return false;
	hit = getIntersectionContext()->m_hitP;
	addHitToSelection();
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

void GLWidget::buildTree()
{
	m_deformer->updateMesh();
	if(m_tree) delete m_tree;
	m_tree = new KdTree;
	m_tree->addMesh(m_mesh);
	m_tree->create();
}

void GLWidget::open()
{
	QString temQStr = QFileDialog::getOpenFileName(this, 
		tr("Open Model File As Temple"), "../", tr("Mesh(*.m)"));
	
	if(temQStr == NULL)
		return;
		
	loadMesh(temQStr.toStdString());
}

void GLWidget::loadMesh(std::string filename)
{
	ESMUtil::Import(filename.c_str(), m_mesh);

	m_deformer->setMesh(m_mesh);
	getSelection()->setTopology(m_deformer->getTopology());
	m_deformer->calculateNormal(m_mesh);
	buildTree();
}

void GLWidget::saveMesh(std::string filename)
{
	ESMUtil::Export(filename.c_str(), m_mesh);
}

void GLWidget::save()
{
	QString temQStr = QFileDialog::getSaveFileName(this, 
		tr("Save Template to Model File"), "./untitled.m", tr("Mesh(*.m)"));
	
	if(temQStr == NULL)
		return;
		
	saveMesh(temQStr.toStdString());
	QMessageBox::information(this, tr("Success"), QString("Template saved as ").append(temQStr));
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	if(e->key() == Qt::Key_A) {
		anchorSelected(1.f);
	}
	else if(e->key() == Qt::Key_D) {
		qDebug() << "deform";
		startDeform();
	}
	else if(e->key() == Qt::Key_F) {
		qDebug() << "fit to target";
		fit();
	}
	else if(e->key() == Qt::Key_Z) {
		removeLastAnchor();
	}

	Base3DView::keyPressEvent(e);
}
//:~
