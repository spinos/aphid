#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "hegWidget.h"
#include <GeoDrawer.h>

using namespace aphid;

namespace ttg {

hegWidget::hegWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
}

hegWidget::~hegWidget()
{}

void hegWidget::clientInit()
{
	//m_scene->init();
	//connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void hegWidget::clientDraw()
{
	updatePerspectiveView();
	getDrawer()->frustum(perspectiveView()->frustum() );
	
	getDrawer()->m_surfaceProfile.apply();
	getDrawer()->setColor(.8f, .8f, .8f);
	
	cvx::Tetrahedron tetra;
	tetra.set(Vector3F(0.f, 1.f, 0.f), 
				Vector3F(-1.f, 0.f, 1.f),
				Vector3F(0.f,-1.f, 0.f), 
				Vector3F(-1.f, 0.f, -1.f) );
				
	cvx::Triangle tri;
	Vector3F vmn;
	
	glBegin(GL_TRIANGLES);
	
	for(int i=0; i< 4; ++i) {
		tetra.getFace(tri, i);
	
		vmn = tri.calculateNormal();
		glNormal3fv((GLfloat *)&vmn );
		glVertex3fv((GLfloat *)tri.p(0) );
		glVertex3fv((GLfloat *)tri.p(1) );		
		glVertex3fv((GLfloat *)tri.p(2) );
	}
	glEnd();
	
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(0.f, 0.f, .5f);
	
	glBegin(GL_TRIANGLES);
	for(int i=0; i< 4; ++i) {
		tetra.getFace(tri, i);
	
		glVertex3fv((GLfloat *)tri.p(0) );
		glVertex3fv((GLfloat *)tri.p(1) );
		glVertex3fv((GLfloat *)tri.p(2) );
	}
	glEnd();
	
	cvx::Hexahedron hexa[4];
	tetra.split(hexa);
	
	getDrawer()->m_surfaceProfile.apply();
	getDrawer()->setColor(.8f, .9f, .8f);
	
	glTranslatef(2.f,0.f,0.f);
	
	cvx::Quad qud;
	for(int i=0; i< 4; ++i) {
		hexa[i].expand(-0.125f);
		for(int j=0; j< 6; ++j) {
			hexa[i].getFace(qud, j);
			
			glBegin(GL_QUADS);
			vmn = qud.calculateNormal();
			glNormal3fv((GLfloat *)&vmn );
			glVertex3fv((GLfloat *)qud.p(0) );
			glVertex3fv((GLfloat *)qud.p(1) );
			glVertex3fv((GLfloat *)qud.p(2) );
			glVertex3fv((GLfloat *)qud.p(3) );
			glEnd();
		}
	}
	
	
}
//! [7]

//! [9]
void hegWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void hegWidget::clientDeselect()
{
}

//! [10]
void hegWidget::clientMouseInput(Vector3F & stir)
{
}

void hegWidget::keyPressEvent(QKeyEvent *e)
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

void hegWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}
	
}