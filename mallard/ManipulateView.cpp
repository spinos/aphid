/*
 *  ManipulateView.cpp
 *  fit
 *
 *  Created by jian zhang on 5/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include <math.h>
#include "ManipulateView.h"
#include <KdTreeDrawer.h>
#include <PatchMesh.h>
#include <KdTree.h>
#include <Ray.h>
#include <ToolContext.h>
#include <TransformManipulator.h>
#include <MeshManipulator.h>
#include <SelectionContext.h>

ManipulateView::ManipulateView(QWidget *parent) : Base3DView(parent)
{
	m_tree = new KdTree;
	m_manipulator = new TransformManipulator;
	m_sculptor = new MeshManipulator;
	m_shouldRebuildTree = false;
	m_selectCtx = new SelectionContext;
}
//! [0]

//! [1]
ManipulateView::~ManipulateView()
{
}

void ManipulateView::clientDraw()
{
/*
	KdTreeDrawer *drawer = getDrawer();
	drawer->hiddenLine(mesh());	
	drawSelection();
	drawAnchors();*/
}
//! [7]

//! [9]
void ManipulateView::clientSelect()
{
}
//! [9]

void ManipulateView::clientDeselect() {}

//! [10]
void ManipulateView::clientMouseInput()
{
	Vector3F hit;
	Ray ray = *getIncidentRay();
	if(interactMode() == ToolContext::SelectVertex) {
		pickupComponent(ray, hit);
	}
}

Vector3F ManipulateView::sceneCenter() const
{
	if(m_tree->isEmpty()) return Base3DView::sceneCenter();
	return m_tree->getBBox().center();
}

bool ManipulateView::pickupComponent(const Ray & ray, Vector3F & hit)
{
	getIntersectionContext()->reset(ray);
	if(!m_tree->intersect(getIntersectionContext())) 
		return false;
	hit = getIntersectionContext()->m_hitP;
	addHitToSelection();
	return true;
}

bool ManipulateView::hitTest(const Ray & ray, Vector3F & hit)
{
	getIntersectionContext()->reset(ray);
	if(!m_tree->intersect(getIntersectionContext())) 
		return false;
	hit = getIntersectionContext()->m_hitP;
	return true;
}

void ManipulateView::selectAround(const Vector3F & center, const float & radius)
{
	m_selectCtx->reset(center, radius);
	m_tree->select(m_selectCtx);
}

void ManipulateView::buildTree()
{
	if(!activeMesh()) return;
	if(m_tree) delete m_tree;
	m_tree = new KdTree;
	m_tree->addMesh(activeMesh());
	m_tree->create();
}

void ManipulateView::focusInEvent(QFocusEvent * event)
{
	if(m_shouldRebuildTree) {
		buildTree();
		m_shouldRebuildTree = false;
	}
	Base3DView::focusInEvent(event);
}

void ManipulateView::setRebuildTree()
{
	m_shouldRebuildTree = true;
}

bool ManipulateView::shouldRebuildTree() const
{
	return m_shouldRebuildTree;
}

KdTree * ManipulateView::getTree() const
{
	return m_tree;
}

PatchMesh * ManipulateView::activeMesh() const
{
	return NULL;
}

void ManipulateView::drawIntersection() const
{
    IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) return;
    
	getDrawer()->drawPrimitivesInNode(m_tree, (KdTreeNode *)ctx->m_cell);
	
	Base3DView::drawIntersection();
}

TransformManipulator * ManipulateView::manipulator()
{
	return m_manipulator;
}

MeshManipulator * ManipulateView::sculptor()
{
    return m_sculptor;
}

void ManipulateView::showManipulator() const
{
	getDrawer()->manipulator(m_manipulator);
}

void ManipulateView::keyPressEvent(QKeyEvent *e)
{	
	switch (e->key()) {
		case Qt::Key_T:
			manipulator()->setToMove();
			break;
		case Qt::Key_R:
			manipulator()->setToRotate();
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(e);
}

void ManipulateView::clearSelection()
{
	m_selectCtx->reset();
	m_manipulator->detach();
	Base3DView::clearSelection();
}

void ManipulateView::processSelection(QMouseEvent *event)
{
	switch (event->button()) {
		case Qt::LeftButton:
			manipulator()->setRotateAxis(TransformManipulator::AY);
			break;
		case Qt::MiddleButton:
			manipulator()->setRotateAxis(TransformManipulator::AZ);
			break;
		default:
			manipulator()->setRotateAxis(TransformManipulator::AX);
			break;
	}
	Base3DView::processSelection(event);
}

void ManipulateView::processDeselection(QMouseEvent * event)
{
	manipulator()->stop();
	Base3DView::processDeselection(event);
}

const std::deque<unsigned> & ManipulateView::selectedQue() const
{
	return m_selectCtx->selectedQue();
}
//:~