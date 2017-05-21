/*
 *  widget.cpp
 *  garden
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <PerspectiveView.h>
#include "widget.h"
#include <GeoDrawer.h>
#include "Vegetation.h"
#include "VegetationPatch.h"
#include "DrawVegetation.h"
#include "gar_common.h"

using namespace aphid;

GLWidget::GLWidget(Vegetation * vege, QWidget *parent) : Base3DView(parent),
m_dspState(gar::dsTriangle)
{
	perspCamera()->setFarClipPlane(30000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(30000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
	m_vege = vege;
	m_vegd = new DrawVegetation;
}

GLWidget::~GLWidget()
{
	delete m_vegd;
}

void GLWidget::clientInit()
{}

void GLWidget::clientDraw()
{
	m_vegd->begin();
	for(int i=0;i<m_vege->numPatches();++i) {
		diffDraw(m_vege->patch(i) );
	}
	m_vegd->end();
}

void GLWidget::diffDraw(VegetationPatch * vgp)
{
	switch(m_dspState) {
		case gar::dsTriangle :
			geomDraw(vgp);
		break;
		case gar::dsDop :
			dopDraw(vgp);
		break;
		case gar::dsPoint :
			pointDraw(vgp );
		break;
		case gar::dsVoxel :
			voxelDraw(vgp);
		break;
		default :
			;
	}
}

void GLWidget::dopDraw(VegetationPatch * vgp)
{
	getDrawer()->setColor(.125f, .125f, .5f);
	getDrawer()->m_surfaceProfile.apply();
	m_vegd->drawDopPatch(vgp);
}

void GLWidget::voxelDraw(VegetationPatch * vgp)
{
	getDrawer()->setColor(.125f, .125f, .5f);
	getDrawer()->m_markerProfile.apply();
	//getDrawer()->m_surfaceProfile.apply();
	m_vegd->drawVoxelPatch(vgp);
}

void GLWidget::pointDraw(VegetationPatch * vgp)
{
	getDrawer()->setColor(.125f, .125f, .5f);
	getDrawer()->m_markerProfile.apply();
	//getDrawer()->m_surfaceProfile.apply();
	m_vegd->drawPointPatch(vgp);
}

void GLWidget::geomDraw(VegetationPatch * vgp)
{
	//getDrawer()->setColor(.125f, .125f, .5f);
	//getDrawer()->m_wireProfile.apply();
	//getDrawer()->m_markerProfile.apply();
	getDrawer()->m_surfaceProfile.apply();
	m_vegd->drawPlantPatch(vgp);
}

void GLWidget::simpleDraw(VegetationPatch * vgp)
{
    getDrawer()->setColor(0.f, .45f, .13f);
	getDrawer()->m_surfaceProfile.apply();
	m_vegd->drawNaive(vgp);
}

void GLWidget::resetPerspViewTransform()
{
static const float mm[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.8660254f, -0.5f, 0.f,
					0.f, 0.5f, 0.8660254f, 0.f,
					0.f, 200.f, 346.4101616f, 1.f};
	Matrix44F mat(mm);
	perspCamera()->setViewTransform(mat, 400.f);
}

void GLWidget::resetOrthoViewTransform()
{
static const float mm1[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.f, -1.f, 0.f,
					0.f, 1.f, 0.f, 0.f,
					0.f, 150.f, 0.f, 1.f};
	Matrix44F mat(mm1);
	orthoCamera()->setViewTransform(mat, 150.f);
	orthoCamera()->setHorizontalAperture(150.f);
}

void GLWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}

void GLWidget::clientDeselect()
{
}

void GLWidget::clientMouseInput(Vector3F & stir)
{
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_W:
			break;
		case Qt::Key_N:
			break;
		default:
			break;
	}
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}
	
void GLWidget::recvToolAction(int x)
{
	qDebug()<<" unknown tool action "<<x;
}

void GLWidget::setDisplayState(int x)
{
	m_dspState = x;
	update();
}
