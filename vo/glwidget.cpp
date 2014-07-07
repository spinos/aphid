#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <Triangle.h>
//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

//! [7]
void GLWidget::clientDraw()
{
    const float bbl = 1.f;
    const float bbh = .67f;
    const int ng = 32;
    const BoundingBox bigBox(bbl, bbl, bbl, bbl + bbh * ng, bbl + bbh * ng, bbl + bbh * ng);
    getDrawer()->setColor(.02f, .6f, .9f);
    getDrawer()->boundingBox(bigBox);
    
    const Vector3F a(3.f, 7.1f, 2.19f);
    const Vector3F b(17.f, 2.45f, 15.19f);
    const Vector3F c(7.f, 17.f, 18.23f);
    const Triangle tri(a, b, c);
    
    getDrawer()->beginWireTriangle();
    getDrawer()->vertex(a);
    getDrawer()->vertex(b);
    getDrawer()->vertex(c);
    getDrawer()->end();
    getDrawer()->setColor(.5f, .5f, .5f);
    
    for(int k = 0; k < ng; k++) {
        for(int j = 0; j < ng; j++) {
            for(int i = 0; i < ng; i++) {
                const BoundingBox bb(bbl + bbh * i, bbl + bbh * j, bbl + bbh * k, bbl + bbh * (i + 1), bbl + bbh * (j + 1), bbl + bbh * (k + 1));
                if(tri.intersect(bb)) 
                    getDrawer()->boundingBox(bb); 
            } 
        }
    }
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

void GLWidget::simulate()
{
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

