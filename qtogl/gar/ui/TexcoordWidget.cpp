/*
 *  widget.cpp
 *  garden
 *
 */
#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <geom/ATriangleMesh.h>
#include "TexcoordWidget.h"
#include "ShrubScene.h"
#include <graphchart/GardenGlyph.h>
#include <attr/PieceAttrib.h>
#include "gar_common.h"

using namespace aphid;

TexcoordWidget::TexcoordWidget(ShrubScene* scene, QWidget *parent) : Base2DView(parent),
m_selectedGlyph(NULL)
{
	orthoCamera()->setFarClipPlane(30000.f);
	orthoCamera()->setNearClipPlane(1.f);
	useOrthoCamera();
	resetView();
	m_scene = scene;
}

TexcoordWidget::~TexcoordWidget()
{}

void TexcoordWidget::clientInit()
{}

void TexcoordWidget::clientDraw()
{
	if(!m_selectedGlyph)
		return;
		
	PieceAttrib* attr = m_selectedGlyph->attrib();
	if(!attr->hasGeom() )
		return;
	
	getDrawer()->m_wireProfile.apply();
	glColor3f(1.f, 1.f, 1.f);
			
	gar::SelectProfile selprof;
	selprof._condition = gar::slIndex;
    
	glEnableClientState(GL_VERTEX_ARRAY);
	
	const int ngeom = attr->numGeomVariations();
    for(int i=0;i<ngeom;++i) {
		selprof._index = i;
        const ATriangleMesh* msh = attr->selectGeom(&selprof);
        drawTexcoord(msh);
    }
	
	glDisableClientState(GL_VERTEX_ARRAY);
}

void TexcoordWidget::resetOrthoViewTransform()
{
static const float mm1[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 1.f, 0.f, 0.f,
					0.f, 0.f, 1.f, 0.f,
					50.f, 50.f, 50.f, 1.f};
	Matrix44F mat(mm1);
	orthoCamera()->setViewTransform(mat, 120.f);
	orthoCamera()->setHorizontalAperture(120.f);
}

void TexcoordWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}

void TexcoordWidget::clientDeselect()
{
}

void TexcoordWidget::clientMouseInput(Vector3F & stir)
{
}

void TexcoordWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_W:
			break;
		case Qt::Key_N:
			break;
		default:
			break;
	}
	Base2DView::keyPressEvent(e);
}

void TexcoordWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base2DView::keyReleaseEvent(event);
}

void TexcoordWidget::recvSelectGlyph(bool x)
{
	if(x) {
		GardenGlyph* g = m_scene->lastSelectedGlyph();
		m_selectedGlyph = g;
	} else {
		m_selectedGlyph = NULL;
	}
	update();
}

void TexcoordWidget::drawTexcoord(const ATriangleMesh* msh)
{
	glPushMatrix();
	glScalef(100.f, 100.f, 1.f);
	
	glBegin(GL_TRIANGLES);
	const int ntv = msh->numTriangles() * 3;
	for(int i=0;i< ntv;++i) {
		const float* tv = &msh->triangleTexcoords()[i*2];
		glVertex3f(tv[0], tv[1], 0.f);
	}
	glEnd();
	
	glPopMatrix();
}
