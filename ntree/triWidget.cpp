/*
 *  TriWidget.cpp
 *  
 *
 *  Created by jian zhang on 3/23/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include "triWidget.h"
#include <BaseCamera.h>
#include <GeoDrawer.h>
#include <NTreeDrawer.h>
#include "GridDrawer.h"

TriWidget::TriWidget(const std::string & filename, QWidget *parent) : 
Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
    m_intersectCtx.m_success = 0;
	
	if(filename.size() > 1) m_container.readTree(filename);
}

TriWidget::~TriWidget()
{}

void TriWidget::clientInit()
{ connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update())); }

void TriWidget::clientDraw()
{
	drawTriangle();
	drawTree();
	drawIntersect();
	drawGrid();
}

void TriWidget::drawTriangle()
{
	if(!m_container.source() ) return;
	
	getDrawer()->m_surfaceProfile.apply();
	getDrawer()->setColor(.8f, .8f, .8f);
		
	const sdb::VectorArray<cvx::Triangle> * src = m_container.source();
	const int n = src->size();
	glBegin(GL_TRIANGLES);
	int i=0;
	for(;i<n;++i) {
		const cvx::Triangle * t = src->get(i);
		
		glNormal3fv((GLfloat *)&t->N(0) );
		glVertex3fv((GLfloat *)t->p(0) );
		
		glNormal3fv((GLfloat *)&t->N(1) );
		glVertex3fv((GLfloat *)t->p(1) );
		
		glNormal3fv((GLfloat *)&t->N(2) );
		glVertex3fv((GLfloat *)t->p(2) );
	}
	glEnd();
}

void TriWidget::drawTree()
{
	if(!m_container.tree() ) return; 
	
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(.15f, .25f, .35f);
	getDrawer()->boundingBox(m_container.tree()->getBBox() );
	
	NTreeDrawer dr;
	dr.drawTree<cvx::Triangle>(m_container.tree() );
}

void TriWidget::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
		case Qt::Key_F:
			qDebug()<<"frame all ";
			camera()->frameAll(m_container.worldBox() );
		    break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}

void TriWidget::clientSelect(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	testIntersect(getIncidentRay());
	setUpdatesEnabled(true);
}

void TriWidget::clientMouseInput(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	testIntersect(getIncidentRay());
	setUpdatesEnabled(true);
}

void TriWidget::testIntersect(const Ray * incident)
{
	m_intersectCtx.reset(*incident);
	if(!m_container.tree() ) return; 
	std::stringstream sst; sst<<incident->m_dir;
	qDebug()<<"interset begin "<<sst.str().c_str();
	m_container.tree()->intersect(&m_intersectCtx);
	qDebug()<<"interset end";
}

void TriWidget::drawIntersect()
{
	Vector3F dst;
	if(m_intersectCtx.m_success) {
		glColor3f(0,1,0);
		dst = m_intersectCtx.m_ray.travel(m_intersectCtx.m_tmax);
	}
	else {
		glColor3f(1,0,0);
		dst = m_intersectCtx.m_ray.destination();
	}
	
	glBegin(GL_LINES);
		glVertex3fv((const GLfloat * )&m_intersectCtx.m_ray.m_origin);
		glVertex3fv((const GLfloat * )&dst);
	glEnd();
	
	BoundingBox b = m_intersectCtx.getBBox();
	b.expand(0.03f);
	getDrawer()->boundingBox(b );
	
	if(m_intersectCtx.m_success) drawActiveSource(m_intersectCtx.m_componentIdx);
}

void TriWidget::drawActiveSource(const unsigned & iLeaf)
{
	if(!m_container.tree() ) return;
	if(!m_container.source() ) return;
	
	glColor3f(0,.6,.4);
	int start, len;
	m_container.tree()->leafPrimStartLength(start, len, iLeaf);
	glBegin(GL_TRIANGLES);
	int i=0;
	for(;i<len;++i) {
		const cvx::Triangle * c = m_container.source()->get( m_container.tree()->primIndirectionAt(start + i) );
		glVertex3fv((const GLfloat * )c->p(0));
		glVertex3fv((const GLfloat * )c->p(1));
		glVertex3fv((const GLfloat * )c->p(2));
	}
	glEnd();
}

void TriWidget::drawGrid()
{
	if(!m_container.grid() ) return;
	
	glColor3f(0,.3,.4);
	GridDrawer dr;
	dr.drawGrid<CartesianGrid>(m_container.grid() );
}
//:~