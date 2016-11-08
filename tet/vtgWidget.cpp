#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <PerspectiveView.h>
#include "vtgWidget.h"
#include <GeoDrawer.h>

using namespace aphid;

namespace ttg {

vtgWidget::vtgWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	
}

vtgWidget::~vtgWidget()
{}

void vtgWidget::clientInit()
{
	cvx::Tetrahedron tetra;
	tetra.set(Vector3F(0.f, 10.f, 0.f), 
				Vector3F(-14.f, 4.f, 10.f),
				Vector3F(0.f,-10.f, 0.f), 
				Vector3F(-10.f, 4.f, -14.f) );
				
	int n = GORDER;
	std::cout << "  N = " << n << "\n";
	
	TetrahedronGridUtil<GORDER> tu4;
	
	int ng = tu4.Ng;
	std::cout << "  Ng = " << ng << "\n";
	
	m_tg = new GridT(tetra);
	
}

void vtgWidget::clientDraw()
{
	updatePerspectiveView();
	//getDrawer()->frustum(perspectiveView()->frustum() );
	
	getDrawer()->m_markerProfile.apply();
	getDrawer()->setColor(.125f, .125f, .5f);
	
	const float nz = .5f;
	const int ng = m_tg->numPoints();
	int i = 0;
	for(;i<ng;++i) {
		getDrawer()->cube(m_tg->pos(i), nz );
			
	}
	
	
}
//! [7]

//! [9]
void vtgWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void vtgWidget::clientDeselect()
{
}

//! [10]
void vtgWidget::clientMouseInput(Vector3F & stir)
{
}

void vtgWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_M:
			//m_scene->progressForward();
			break;
		case Qt::Key_N:
			//m_scene->progressBackward();
			break;
		default:
			break;
	}
	Base3DView::keyPressEvent(e);
}

void vtgWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}
	
}