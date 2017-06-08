/*
 *  widget.cpp
 *  hes viewer
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <PerspectiveView.h>
#include <HesScene.h>
#include <geom/ATriangleMeshGroup.h>
#include "glWidget.h"

using namespace aphid;

GLWidget::GLWidget(const aphid::HesScene* scene, QWidget *parent) : Base3DView(parent),
m_scene(scene),
m_dspState(0)
{
	perspCamera()->setFarClipPlane(1000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(1000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
	glDisable(GL_POLYGON_OFFSET_LINE);
	glEnable(GL_POLYGON_OFFSET_FILL);
}

void GLWidget::clientDraw()
{
	const int nmsh = m_scene->numMeshes();
	for(int i=0;i<nmsh;++i) {
		drawMesh(m_scene->mesh(i) );
	}
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
    int camerop = HesScene::opUnknown;
	switch (e->key()) {
		case Qt::Key_F:
		    camerop = HesScene::opFrameAll;
			break;
		case Qt::Key_N:
			break;
		default:
			break;
	}
	
	processSceneCamera(camerop);
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

void GLWidget::drawMesh(const ATriangleMeshGroup* msh)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)msh->points() );
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor3f(.2f, .2f, .2f);
	glDrawElements(GL_TRIANGLES, msh->numIndices(), GL_UNSIGNED_INT, msh->indices() );
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPolygonOffset(1.0f, 1.0f);
	glColor3f(.9f, .9f, .9f);
	glDrawElements(GL_TRIANGLES, msh->numIndices(), GL_UNSIGNED_INT, msh->indices() );
	
	glDisableClientState(GL_VERTEX_ARRAY);
}

void GLWidget::processSceneCamera(int x)
{
    if(x == HesScene::opUnknown ) {
        return;
    }
    
    if(x == HesScene::opFrameAll) {
        const BoundingBox bbx = m_scene->calculateBBox();
        viewAll(bbx);
    }
    
}

