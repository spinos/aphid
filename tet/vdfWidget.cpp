#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "vdfWidget.h"
#include <GeoDrawer.h>

using namespace aphid;

namespace ttg {

vdfWidget::vdfWidget(ttg::Scene * sc, QWidget *parent) : Base3DView(parent)
{
	m_scene = sc;
	m_scene->setView(perspectiveView() );
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
}

vdfWidget::~vdfWidget()
{}

void vdfWidget::clientInit()
{
	m_scene->init();
	//connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void vdfWidget::clientDraw()
{
	updatePerspectiveView();
	getDrawer()->frustum(perspectiveView()->frustum() );
	m_scene->draw(getDrawer() );
}
//! [7]

//! [9]
void vdfWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void vdfWidget::clientDeselect()
{
}

//! [10]
void vdfWidget::clientMouseInput(Vector3F & stir)
{
}

void vdfWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_M:
			m_scene->progressForward();
			break;
		case Qt::Key_N:
			m_scene->progressBackward();
			break;
		default:
			break;
	}
	Base3DView::keyPressEvent(e);
}

void vdfWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}
	
}