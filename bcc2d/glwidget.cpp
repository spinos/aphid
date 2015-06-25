#include <QtGui>
#include "BccGlobal.h"
#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "DrawNp.h"
#include "BccWorld.h"
#include "FitTest.h"

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
#if TEST_FIT
    FitBccMeshBuilder::EstimatedGroupSize = 2.1f;
	m_fit = new FitTest(getDrawer());
#else
	m_world = new BccWorld(getDrawer());
#endif
}

GLWidget::~GLWidget()
{ 
}

void GLWidget::clientInit()
{
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
#if TEST_FIT
	m_fit->draw();
#else
	m_world->draw();
	std::stringstream sst;
	sst.str("");
	sst<<"n curves: "<<m_world->numCurves();
	hudText(sst.str(), 1);
	sst.str("");
	sst<<"n tetrahedrons: "<<m_world->numTetrahedrons();
    hudText(sst.str(), 2);
	sst.str("");
	sst<<"n points: "<<m_world->numPoints();
    hudText(sst.str(), 3);
#endif
}

void GLWidget::clientSelect(QMouseEvent * event)
{
	setUpdatesEnabled(false);
#if TEST_FIT

#else
	m_world->select(getIncidentRay());
#endif
	setUpdatesEnabled(true);
}

void GLWidget::clientDeselect(QMouseEvent * event) 
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientMouseInput(QMouseEvent * event)
{
	setUpdatesEnabled(false);
#if TEST_FIT

#else
	m_world->select(getIncidentRay());
#endif	
	setUpdatesEnabled(true);
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
	if(event->modifiers() == Qt::ControlModifier | Qt::MetaModifier) {
		if(event->key() == Qt::Key_S) {
#if TEST_FIT

#else
			m_world->save();
#endif
		}
	}
    setUpdatesEnabled(false);	
	switch (event->key()) {
		case Qt::Key_K:
		    m_world->rebuildTetrahedronsMesh(-1.f);
			break;
		case Qt::Key_L:
		    m_world->rebuildTetrahedronsMesh(1.f);
			break;
		case Qt::Key_M:
		    m_world->reduceSelected(.13f);
			break;
		/*case Qt::Key_S:
		    m_world->moveTestP(0.f, -.1f, 0.f);
			break;
		case Qt::Key_F:
		    m_world->moveTestP(0.f, 0.f, .1f);
			break;
		case Qt::Key_B:
		    m_world->moveTestP(0.f, 0.f, -.1f);
			break;*/
		default:
			break;
	}
	setUpdatesEnabled(true);
	Base3DView::keyPressEvent(event);
}
