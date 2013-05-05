/*
 *  glwidget.cpp
 *  fit
 *
 *  Created by jian zhang on 5/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include <QtOpenGL>
#include "glwidget.h"
#include "FitDeformer.h"
	
//! [0]
GLWidget::GLWidget(QWidget *parent) : SingleModelView(parent)
{
	m_deformer = new FitDeformer;
	m_deformer->setAnchors(m_anchors);
	
	QString filename  = QString("%1/mdl/face.m")
                 .arg(QDir::currentPath());

	loadMesh(filename.toStdString().c_str());
}
//! [0]

//! [1]
GLWidget::~GLWidget() {}

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
    getSelection()->enableVertexPath();
    
	SingleModelView::clientSelect(origin, displacement, hit);
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

void GLWidget::buildTree()
{
	m_deformer->updateMesh();
	SingleModelView::buildTree();
}

void GLWidget::loadMesh(std::string filename)
{
	SingleModelView::loadMesh(filename);

	m_deformer->setMesh(m_mesh);
	getSelection()->setTopology(m_deformer->getTopology());
	m_deformer->calculateNormal(m_mesh);
	buildTree();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	if(e->key() == Qt::Key_F) {
		qDebug() << "fit to target";
		fit();
	}
	else if(e->key() == Qt::Key_S) {
		qDebug() << "smooth by 0.9";
		m_deformer->setSmoothFactor(m_deformer->getSmoothFactor() * 0.9f);
		startDeform();
	}

	SingleModelView::keyPressEvent(e);
}
//:~
