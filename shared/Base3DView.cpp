
#include <QtGui>
#include <QtOpenGL>

#include "Base3DView.h"


#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

//! [0]
Base3DView::Base3DView(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    m_backgroundColor = QColor::fromCmykF(0.29, 0.29, 0.20, 0.0);
	
	fCamera = new BaseCamera();
}
//! [0]

//! [1]
Base3DView::~Base3DView()
{
    delete fCamera;
}
//! [1]

//! [2]
QSize Base3DView::minimumSizeHint() const
{
    return QSize(50, 50);
}
//! [2]

//! [3]
QSize Base3DView::sizeHint() const
//! [3] //! [4]
{
    return QSize(640, 480);
}
//! [4]

//! [6]
void Base3DView::initializeGL()
{
    qglClearColor(m_backgroundColor.dark());

    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);
    //glEnable(GL_LIGHTING);
    //glEnable(GL_LIGHT0);
    glEnable(GL_MULTISAMPLE);
    //static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
    //glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
		
}
//! [6]

//! [7]
void Base3DView::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float m[16];
	fCamera->getMatrix(m);
	glMultMatrixf(m);
	clientDraw();
	glFlush();
}
//! [7]

//! [8]
void Base3DView::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
	fCamera->setPortWidth(width);
	fCamera->setPortHeight(height);
	if(fCamera->isOrthographic())
		updateOrthoProjection();
}
//! [8]

//! [9]
void Base3DView::mousePressEvent(QMouseEvent *event)
{
    m_lastPos = event->pos();
    if(event->modifiers() == Qt::AltModifier) 
        return;
    
    processSelection(event);
}
//! [9]

void Base3DView::mouseReleaseEvent(QMouseEvent *event)
{
    processDeselection(event);
}

//! [10]
void Base3DView::mouseMoveEvent(QMouseEvent *event)
{
    if(event->modifiers() == Qt::AltModifier)
        processCamera(event);
    else 
        processMouseInput(event);

    m_lastPos = event->pos();
}
//! [10]

void Base3DView::processCamera(QMouseEvent *event)
{
    int dx = event->x() - m_lastPos.x();
    int dy = event->y() - m_lastPos.y();
    if (event->buttons() & Qt::LeftButton) {
        fCamera->tumble(dx, dy);
    } 
	else if (event->buttons() & Qt::MidButton) {
		fCamera->track(dx, dy);
    }
	else if (event->buttons() & Qt::RightButton) {
		fCamera->zoom(dy);
		if(fCamera->isOrthographic())
			updateOrthoProjection();
    }
}

void Base3DView::processSelection(QMouseEvent *event)
{
    Vector3F origin, incident;
    fCamera->incidentRay(event->x(), event->y(), origin, incident);
    incident = incident.normal() * 1000.f;
    clientSelect(origin, incident, m_hitPosition);
}

void Base3DView::processDeselection(QMouseEvent *event)
{
    clientDeselect();
}

void Base3DView::processMouseInput(QMouseEvent *event)
{
    fCamera->intersection(event->x(), event->y(), m_hitPosition);
    int dx = event->x() - m_lastPos.x();
    int dy = event->y() - m_lastPos.y();
    Vector3F injv;
    fCamera->screenToWorld(dx, dy, injv);
    clientMouseInput(injv);
}

void Base3DView::clientDraw()
{
    
}

void Base3DView::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
    
}

void Base3DView::clientDeselect()
{
    
}

void Base3DView::clientMouseInput(Vector3F & stir)
{
    
}

void Base3DView::updateOrthoProjection()
{
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	
	float aspect = fCamera->aspectRatio();
	float fov = fCamera->getHorizontalAperture();
	float right = fov/ 2.f;
	float top = right / aspect;

    glOrtho(-right, right, -top, top, 1.0, 1000.0);

    glMatrixMode(GL_MODELVIEW);
}
