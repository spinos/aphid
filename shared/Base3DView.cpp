
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
	m_orthoCamera = new BaseCamera;
	m_perspCamera = new PerspectiveCamera;
	fCamera = m_orthoCamera;
	m_drawer = new KdTreeDrawer;
	m_selected = new SelectionArray;
	m_selected->setComponentFilterType(PrimitiveFilter::TVertex);
	m_intersectCtx = new IntersectionContext;
	m_intersectCtx->setComponentFilterType(PrimitiveFilter::TVertex);
	m_timer = new QTimer(this);
	connect(m_timer, SIGNAL(timeout()), this, SLOT(update()));
	m_timer->start(30);
	setFocusPolicy(Qt::ClickFocus);
	m_isFocused = 0;
}
//! [0]

//! [1]
Base3DView::~Base3DView()
{
	delete fCamera;
	delete m_drawer;
	delete m_selected;
	delete m_intersectCtx;
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

BaseCamera * Base3DView::getCamera() const
{
	return fCamera;
}

KdTreeDrawer * Base3DView::getDrawer() const
{
	return m_drawer;
}

SelectionArray * Base3DView::getSelection() const
{
	return m_selected;
}

IntersectionContext * Base3DView::getIntersectionContext() const
{
	return m_intersectCtx;
}

void Base3DView::initializeGL()
{
    qglClearColor(m_backgroundColor.dark());

    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);
	glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
	glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST );
    //glEnable(GL_LIGHTING);
    //glEnable(GL_LIGHT0);
    glEnable(GL_MULTISAMPLE);
    //static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
    //glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
	glDepthFunc(GL_LEQUAL);	
}
//! [6]

//! [7]
void Base3DView::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float m[16];
	getCamera()->getMatrix(m);
	glMultMatrixf(m);
	clientDraw();
	if(m_isFocused) {
		Vector3F corners[4];
		getCamera()->frameCorners(corners[0], corners[1], corners[2], corners[3]);
		glColor3f(0.f, 0.f, 1.f);
		glBegin(GL_LINE_LOOP);
		glVertex3f(corners[0].x, corners[0].y, corners[0].z);
		glVertex3f(corners[1].x, corners[1].y, corners[1].z);
		glVertex3f(corners[2].x, corners[2].y, corners[2].z);
		glVertex3f(corners[3].x, corners[3].y, corners[3].z);
		glEnd();
	}
	glFlush();
}
//! [7]

//! [8]
void Base3DView::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
	m_perspCamera->setPortWidth(width);
	m_perspCamera->setPortHeight(height);
	m_orthoCamera->setPortWidth(width);
	m_orthoCamera->setPortHeight(height);
	if(getCamera()->isOrthographic())
		updateOrthoProjection();
	else
		updatePerspProjection();
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
        getCamera()->tumble(dx, dy);
    } 
	else if (event->buttons() & Qt::MidButton) {
		getCamera()->track(dx, dy);
    }
	else if (event->buttons() & Qt::RightButton) {
		getCamera()->zoom(-dx / 2 + -dy / 2);
		if(getCamera()->isOrthographic())
			updateOrthoProjection();
		else
			updatePerspProjection();
    }
}

void Base3DView::processSelection(QMouseEvent *event)
{
    Vector3F origin, incident;
    getCamera()->incidentRay(event->x(), event->y(), origin, incident);
    incident = incident.normal() * 1000.f;
	
    if(event->modifiers() == Qt::ShiftModifier) 
		m_selected->disableVertexPath();
	else 
		m_selected->enableVertexPath();
		
	Vector3F nouse;
    clientSelect(origin, incident, nouse);
}

void Base3DView::processDeselection(QMouseEvent *event)
{
    clientDeselect();
}

void Base3DView::processMouseInput(QMouseEvent *event)
{
    int dx = event->x() - m_lastPos.x();
    int dy = event->y() - m_lastPos.y();
    Vector3F injv;
    getCamera()->screenToWorldVector(dx, dy, injv);
	Vector3F origin, incident;
    getCamera()->incidentRay(event->x(), event->y(), origin, incident);
    incident = incident.normal() * 1000.f;
    clientMouseInput(origin, incident, injv);
}

void Base3DView::clientDraw()
{
    
}

void Base3DView::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
    
}

void Base3DView::clientDeselect()
{
    
}

void Base3DView::clientMouseInput(Vector3F & origin, Vector3F & displacement, Vector3F & stir)
{
    
}

void Base3DView::sceneCenter(Vector3F & dst) const
{
    dst.x = dst.y = 0.f;
    dst.z = -1.f;
}

void Base3DView::updateOrthoProjection()
{
	makeCurrent();
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	
	float aspect = getCamera()->aspectRatio();
	float fov = getCamera()->fieldOfView();
	float right = fov/ 2.f;
	float top = right / aspect;

    glOrtho(-right, right, -top, top, 1.0, 1000.0);

    glMatrixMode(GL_MODELVIEW);
	doneCurrent();
}

void Base3DView::updatePerspProjection()
{
	makeCurrent();
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	
	GLdouble left,right,bottom,top;
		
	right = getCamera()->frameWidth() * 0.5f;
	left = -right;
	top = getCamera()->frameHeight() * 0.5f;
	bottom = -top;

    glFrustum(left, right, bottom, top, 1.0, 1000.0);
	
    glMatrixMode(GL_MODELVIEW);
	doneCurrent();
}

void Base3DView::resetView()
{
	getCamera()->reset();
	if(getCamera()->isOrthographic())
		updateOrthoProjection();
	else
		updatePerspProjection();
}

void Base3DView::drawSelection()
{
	m_drawer->setColor(0.f, .8f, .2f);
	m_drawer->components(m_selected);
}

void Base3DView::clearSelection()
{
	m_selected->reset();
}

void Base3DView::addHitToSelection()
{
	m_selected->add(m_intersectCtx->m_geometry, m_intersectCtx->m_componentIdx, m_intersectCtx->m_hitP);
}

void Base3DView::growSelection()
{
	m_selected->grow();
}

void Base3DView::shrinkSelection()
{
	m_selected->shrink();
}

void Base3DView::frameAll()
{
    Vector3F coi;
    sceneCenter(coi);
    Vector3F eye = coi + Vector3F(0.f, 0.f, 100.f);
    
    getCamera()->lookFromTo(eye, coi);
}

void Base3DView::keyPressEvent(QKeyEvent *e)
{
	if(e->key() == Qt::Key_Space) {
		qDebug() << "clear selection";
		clearSelection();
	}
	else if(e->key() == Qt::Key_H) {
		qDebug() << "reset camera";
		resetView();
	}
	else if(e->key() == Qt::Key_BracketRight) {
		growSelection();
	}
	else if(e->key() == Qt::Key_BracketLeft) {
		shrinkSelection();
	}
	else if(e->key() == Qt::Key_Up) {
		getCamera()->moveForward(23);
	}
	else if(e->key() == Qt::Key_Down) {
		getCamera()->moveForward(-23);
	}
	else if(e->key() == Qt::Key_O) {
		if(getCamera()->isOrthographic()) {
			fCamera = m_perspCamera;
			fCamera->copyTransformFrom(m_orthoCamera);
			updatePerspProjection();
		}
		else {
			fCamera = m_orthoCamera;
			fCamera->copyTransformFrom(m_perspCamera);
			updateOrthoProjection();
		}
	}
    else if(e->key() == Qt::Key_G) {
		qDebug() << "frame all camera";
		frameAll();
	}
	
	QWidget::keyPressEvent(e);
}

void Base3DView::focusInEvent(QFocusEvent * event)
{
	m_isFocused = 1;
	m_timer->start(30);
	QGLWidget::focusInEvent(event);
}

void Base3DView::focusOutEvent(QFocusEvent * event)
{
	m_isFocused = 0;
	m_timer->stop();
	QGLWidget::focusOutEvent(event);
}
//:~
