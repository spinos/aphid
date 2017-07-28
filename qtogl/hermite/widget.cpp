#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <math/linspace.h>
#include <ogl/DrawArrow.h>
#include <ogl/TranslationHandle.h>
#include <BaseCamera.h>
#include <pbd/Beam.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
static const float P0[4][3] = {
{1.4973, 0.0135717, 0},
{0.41143, 9.21691, -0.852699},
{-0.411491, 17.563, -2.99834},
{-0.30176, 21.7558, -8.07755}
};

static const float T0[4][3] = {
{1.16552, 6.52755, -0.949179},
{-1.9231, 6.77553, -0.236556},
{2.79723, 6.54754, -3.41737},
{-5.89213, 1.43714, -8.56933}
};	
	m_incident = new Ray;
	m_space = new Matrix44F;
	m_tranh = new TranslationHandle(m_space);
    m_tranh->setRadius(4.f);
	m_pntI = 0;
	m_tngSel = false;
	
	m_beam = new pbd::Beam;
	for(int i=0;i<3;++i) {
		m_beam->setPieceBegin(i, P0[i], T0[i]);
		m_beam->setPieceEnd(i, P0[i+1], T0[i+1]);
	}
/// 5 seg per piece
	m_beam->createNumSegments(5);
		
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{
	//getDrawer()->m_wireProfile.apply();
	getDrawer()->m_markerProfile.apply();

	getDrawer()->setColor(1.f, 1.f, .6f);

	const int& ns = m_beam->numSegments();
	glBegin(GL_LINES);
	for(int i=0;i<ns;++i) {
		glVertex3fv((const float *)&m_beam->getParticlePnt(i) );
		glVertex3fv((const float *)&m_beam->getParticlePnt(i+1) );
	}
	glEnd();
/// material frame x
	getDrawer()->setColor(1.f, 0.f, 0.f);
	glBegin(GL_LINES);
	for(int i=0;i<ns;++i) {
		glVertex3fv((const float *)&m_beam->getSegmentMidPnt(i) );
		glVertex3fv((const float *)&m_beam->getGhostParticlePnt(i) );
	}
	glEnd();
	
	getDrawer()->setColor(0.f, 0.f, 1.f);
/// control	
	for(int j=0;j<6;++j) {
		Vector3F pj = m_beam->Pnt(j);
		Vector3F p1j = pj + m_beam->Tng(j);
		getDrawer()->arrow(pj, p1j);
	}
	
	Vector3F veye = getCamera()->eyeDirection();
	Matrix44F meye = *m_space;
	meye.setFrontOrientation(veye );

	m_tranh->draw(&meye);
	
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	m_tranh->begin(getIncidentRay() );
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	m_tranh->end();
	update();
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	m_tranh->translate(getIncidentRay() );
	
	if(m_tngSel) {
		moveTng();
	} else {
		movePnt();
	}
	
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_K:
			selectSeg(-1);
			break;
		case Qt::Key_L:
			selectSeg(1);
			break;
		case Qt::Key_I:
			toggleTngSel();
			break;
		case Qt::Key_P:
			printSegs();
			break;
		default:
			break;
	}
	Base3DView::keyPressEvent(e);
}

void GLWidget::selectSeg(int d)
{
	m_pntI += d;
	if(m_pntI < 0) m_pntI = 3;
	if(m_pntI > 3) m_pntI = 0;
	
	if(m_tngSel) {
		if(m_pntI == 3)
			m_space->setTranslation(m_beam->Pnt(5) + m_beam->Tng(5) );
		else
			m_space->setTranslation(m_beam->Pnt(m_pntI * 2) + m_beam->Tng(m_pntI * 2) );
			
	} else {
		if(m_pntI == 3)
			m_space->setTranslation(m_beam->Pnt(5) );
		else
			m_space->setTranslation(m_beam->Pnt(m_pntI * 2) );
	}
	update();
}

void GLWidget::toggleTngSel()
{ 
	m_tngSel = !m_tngSel; 
	selectSeg(0);
}

void GLWidget::movePnt()
{	
	if(m_pntI == 0) {
		Vector3F p1 = m_beam->Tng(0);
		m_beam->setPieceBegin(0, m_space->getTranslation(), p1);
	} else if(m_pntI == 3) {
		Vector3F p1 = m_beam->Tng(5);
		m_beam->setPieceEnd(2, m_space->getTranslation(), p1);
	} else {
		Vector3F p1 = m_beam->Tng(m_pntI * 2);
		m_beam->setPieceBegin(m_pntI, m_space->getTranslation(), p1);
		p1 = m_beam->Tng(m_pntI * 2 - 1);
		m_beam->setPieceEnd(m_pntI-1, m_space->getTranslation(), p1);
	}
	m_beam->calculatePnts();
}

void GLWidget::moveTng()
{	
	if(m_pntI == 0) {
		Vector3F p1 = m_beam->Pnt(0);
		m_beam->setPieceBegin(0, p1, m_space->getTranslation() - p1);
	} else if(m_pntI == 3) {
		Vector3F p1 = m_beam->Pnt(5);
		m_beam->setPieceEnd(2, p1, m_space->getTranslation() - p1);
	} else {
		Vector3F p1 = m_beam->Pnt(m_pntI * 2);
		m_beam->setPieceBegin(m_pntI, p1, m_space->getTranslation() - p1);
		p1 = m_beam->Pnt(m_pntI * 2 - 1);
		m_beam->setPieceEnd(m_pntI-1, p1, m_space->getTranslation() - p1);
	}
	m_beam->calculatePnts();
}

void GLWidget::printSegs()
{
	Vector3F tmp;
	std::cout<<"\n P ";
	for(int i=0;i<3;++i) {
		tmp = m_beam->Pnt(i * 2);
		std::cout<<"\n {"<<tmp.x<<","<<tmp.y<<","<<tmp.z<<"},";
	}
	
	tmp = m_beam->Pnt(5);
	std::cout<<"\n {"<<tmp.x<<","<<tmp.y<<","<<tmp.z<<"},";
	
	std::cout<<"\n T ";
	for(int i=0;i<3;++i) {
		tmp = m_beam->Tng(i * 2);
		std::cout<<"\n {"<<tmp.x<<","<<tmp.y<<","<<tmp.z<<"},";
	}
	tmp = m_beam->Tng(5);
	std::cout<<"\n {"<<tmp.x<<","<<tmp.y<<","<<tmp.z<<"},";
	std::cout.flush();
	
}
