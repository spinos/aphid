/* 
 *  slerp test
 */
 
#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <math/linspace.h>
#include <math/Quaternion.h>
#include <math/SplineMap1D.h>
#include <ogl/DrawArrow.h>
#include <ogl/TranslationHandle.h>
#include <ogl/RotationHandle.h>
#include <ogl/DrawArrow.h>
#include <BaseCamera.h>
#include <geom/FiberBundle.h>
#include "slerp_common.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	m_incident = new Ray;
	m_space = new Matrix44F;
	m_tranh = new TranslationHandle(m_space);
    m_tranh->setRadius(4.f);
	m_roth = new RotationHandle(m_space);
	m_roth->setRadius(3.2f);
	
	m_fbbld = new FiberBundleBuilder;
	m_fbbld->begin();
	m_fbbld->addPoint(Vector3F(0,0,0) );
	m_fbbld->addPoint(Vector3F(0,10,0) );
	m_fbbld->addPoint(Vector3F(8,20,0) );
	m_fbbld->addStrand();
	m_fbbld->addPoint(Vector3F(-1,0,0) );
	m_fbbld->addPoint(Vector3F(-3,10,0) );
	m_fbbld->addPoint(Vector3F(-8,20,0) );
	m_fbbld->addPoint(Vector3F(-11,30,0) );
	m_fbbld->end();
	
	m_fib = new FiberBundle;
	m_fib->create(*m_fbbld);
	
	m_mode = slp::ctUnknown; 
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{	
	if(m_mode == slp::ctNew) {
		drawBuilder();
	} else {
		drawPoints();
		drawInterp();
	}
	
	getDrawer()->m_markerProfile.apply();
	getDrawer()->setColor(1.f, 1.f, .6f);

	Vector3F veye = getCamera()->eyeDirection();
	Matrix44F meye = *m_space;
	meye.setFrontOrientation(veye );

	switch(m_mode) {
		case slp::ctMove:
		case slp::ctMoveStrand:
			m_tranh->draw(&meye);
		break;
		case slp::ctRotate:
		case slp::ctRotateStrand:
			m_roth->draw(&meye);
		break;
		default:
		;
	}
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	const Ray* incr = getIncidentRay();
	selectPoint(incr );
	switch(m_mode) {
		case slp::ctMove:
		case slp::ctMoveStrand:
			m_tranh->begin(incr );
		break;
		case slp::ctRotate:
		case slp::ctRotateStrand:
			m_roth->begin(incr );
		break;
		default:
		;
	}
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	switch(m_mode) {
		case slp::ctMove:
		case slp::ctMoveStrand:
		m_tranh->end();
		break;
		case slp::ctRotate:
		case slp::ctRotateStrand:
		m_roth->end();
		break;
		default:
		;
	}
	update();
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	switch(m_mode) {
		case slp::ctMove:
		m_tranh->translate(getIncidentRay() );
		movePoint();
		break;
		case slp::ctMoveStrand:
		m_tranh->translate(getIncidentRay() );
		moveStrand();
		break;
		case slp::ctRotate:
		m_roth->rotate(getIncidentRay() );
		rotatePoint();
		break;
		case slp::ctRotateStrand:
		m_roth->rotate(getIncidentRay() );
		rotateStrand();
		break;
		default:
		;
	}
	m_fib->update();
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_E:
			m_mode = slp::ctRotate;
			break;
		case Qt::Key_R:
			m_mode = slp::ctRotateStrand;
			break;
		case Qt::Key_W:
			m_mode = slp::ctMove;
			break;
		case Qt::Key_T:
			m_mode = slp::ctMoveStrand;
			break;
		case Qt::Key_I:
			m_fib->initialize();
			break;
		case Qt::Key_P:
			break;
		default:
		;
	}
	Base3DView::keyPressEvent(e);
}

void GLWidget::selectPoint(const aphid::Ray* incident)
{
	float minD = 1e9f;
	bool stat = m_fib->selectPoint(minD, incident);
	if(!stat)
		return;
	
	m_fib->getSelectPointSpace(m_space);
}

void GLWidget::drawPoints()
{
	getDrawer()->m_surfaceProfile.apply();
	
	BoundingBox ba(-1,-1,-1,1,1,1);
	DrawBox dba;
	dba.updatePoints(&ba);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	Matrix44F mat;
	
	for(int i=0;i<m_fib->numPoints();++i) {	
	
		const OrientedPoint& pv = m_fib->points()[i];
		
		mat.setRotation(pv._q);
		mat.setTranslation(pv._x);
		
		glPushMatrix();
		getDrawer()->useSpace(mat);
		
		dba.drawASolidBox();
		
		glPopMatrix();
	}
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
}

void GLWidget::drawInterp()
{
	getDrawer()->m_wireProfile.apply();
	
	DrawArrow darr;
	darr.begin();
	
	const int& n = m_fib->numStrands();
	for(int i=0;i<n;++i)
		drawFiberInterp(m_fib->strand(i), &darr);
	
	darr.end();
}

void GLWidget::drawFiberInterp(const aphid::Fiber* fib,
				aphid::DrawArrow* darr)
{
	const int nj = fib->numSegments();
	Matrix44F mat;
	float segl;
	
	const float dp = .125f;
	for(int j=0;j<nj;++j) {
		const FiberUnit& uj = fib->segments()[j];
		
		for(int i=0;i<8;++i) {
	
			Fiber::InterpolateSpace(mat, segl, uj, dp*i);
			mat.scaleBy(Vector3F(segl, .5f, .5f) );
			darr->drawBoneArrowAt(&mat);
		
		}
	}
}

void GLWidget::movePoint()
{
	Vector3F dv;
	m_tranh->getDeltaTranslation(dv);
	m_fib->moveSelectedPoint(dv);
}

void GLWidget::rotatePoint()
{
	Matrix33F rot;
	m_roth->getDeltaRotation(rot);
	Quaternion dq;
	rot.getQuaternion(dq);
	m_fib->rotateSelectedPoint(dq);
}

void GLWidget::moveStrand()
{
	Vector3F dv;
	m_tranh->getDeltaTranslation(dv);
	m_fib->moveSelectedStrand(dv);
}

void GLWidget::rotateStrand()
{
	Matrix33F rot;
	m_roth->getDeltaRotation(rot);
	Quaternion dq;
	rot.getQuaternion(dq);
	m_fib->rotateSelectedStrand(dq);
}

void GLWidget::drawBuilder()
{
	getDrawer()->m_surfaceProfile.apply();
	
	for(int j=0;j<m_fbbld->numStrands();++j) {
		const FiberBulder* sj = m_fbbld->strand(j);
		for(int i=1;i<sj->numPoints();++i) {
		}
	}
}

void GLWidget::recvToolSelected(int x)
{
	QString toolName("unknown");
	switch (x) {
		case slp::ctNew:
			toolName = "new curve";
			m_mode = slp::ctNew;
			m_fib->dump(m_fbbld);
			break;
		case slp::ctMove:
			toolName = "move";
			m_mode = slp::ctMove;
			break;
		case slp::ctMoveStrand:
			toolName = "move strand";
			m_mode = slp::ctMoveStrand;
			break;
		case slp::ctRotateStrand:
			toolName = "rotate strand";
			m_mode = slp::ctRotateStrand;
			break;
		default:
			m_mode = slp::ctUnknown;
			break;
	}
	qDebug()<<" select tool "<<toolName;
	if(m_mode > slp::ctUnknown)
		update();
}
