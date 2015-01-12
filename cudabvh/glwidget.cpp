#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"

#include <KdTreeDrawer.h>
#include <CUDABuffer.h>
#include <BvhSolver.h>
#include "bvh_common.h"
#include <radixsort_implement.h>
#include <app_define.h>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_solver = new BvhSolver;
	m_displayLevel = 0;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	m_solver->init();
	//m_cvs->create(m_curve->numVertices() * 12);
	//m_cvs->hostToDevice(m_curve->m_cvs, m_curve->numVertices() * 12);
	//m_program->run(m_vertexBuffer, m_cvs, m_curve);
	connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
	// connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_solver->displayVertex());
	glDrawElements(GL_TRIANGLES, m_solver->getNumTriangleFaceVertices(), GL_UNSIGNED_INT, m_solver->getIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
	
	// showEdgeContacts();
	showAabbs();
	m_solver->setAlpha((float)elapsedTime()/300.f);
	// qDebug()<<"drawn in "<<deltaTime();
}

void GLWidget::showEdgeContacts()
{
    glPolygonMode(GL_FRONT, GL_FILL);
    glPolygonMode(GL_BACK, GL_LINE);
    
    float * dsyV = m_solver->displayVertex();
	EdgeContact * ec = m_solver->edgeContacts();
	unsigned ne = m_solver->numEdges();
	unsigned a, b, c, d;
	const float h = 0.2f;
	const unsigned maxI = m_solver->numVertices();
	float * p;
	glBegin(GL_TRIANGLES);
	for(unsigned i=0; i < ne; i++) {
	    EdgeContact & ae = ec[i];
	    a = ae.v[0];
	    b = ae.v[1];
	    c = ae.v[2];
	    d = ae.v[3];
	    
	    if(c < maxI && d < maxI) {
	        p = &dsyV[a * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[b * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[c * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        
	        p = &dsyV[b * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[a * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[d * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	    }
	    else if(c < maxI) {
	        p = &dsyV[a * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[b * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[c * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	    }
	    else if(d < maxI) {
	        p = &dsyV[a * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[b * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[d * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	    }
	}
	glEnd();
}
void GLWidget::showAabbs()
{
	Aabb ab = m_solver->combinedAabb();
	GeoDrawer * dr = getDrawer();
    BoundingBox bb; 
	bb.setMin(ab.low.x, ab.low.y, ab.low.z);
	bb.setMax(ab.high.x, ab.high.y, ab.high.z);
	glColor3f(0.1f, 0.4f, 0.3f);
    // dr->boundingBox(bb);
	
#ifdef BVHSOLVER_DBG_DRAW
	
#ifdef BVHSOLVER_DBG_DRAW_INTERNALBOX
	Aabb * boxes = m_solver->displayInternalAabbs();
	int * levels = m_solver->displayInternalDistances();
    unsigned ne = m_solver->numInternalNodes();

	int ninvalidbox = 0;
    for(unsigned i=0; i < ne; i++) {
		if(levels[i] >  m_displayLevel) continue;
        ab = boxes[i];
        
		bb.setMin(ab.low.x, ab.low.y, ab.low.z);
        bb.setMax(ab.high.x, ab.high.y, ab.high.z);
		
		if(!bb.isValid() || bb.area() < 0.1f) {
			// qDebug()<<bb.str().c_str();
			ninvalidbox++;
		}
		
		float redc = ((float)(levels[i] % 22))/22.f;
		
		glColor3f(redc, 1.f - redc, 0.f);
	
        dr->boundingBox(bb);
    }
	if(ninvalidbox > 0) qDebug()<<"n invalid box "<<ninvalidbox;
#endif	
/*
#ifdef BVHSOLVER_DBG_DRAW_LEAFHASH
	KeyValuePair * leafHash = m_solver->displayLeafHash();
	glColor3f(0.8f, 0.1f, 0.f);
	glBegin(GL_LINES);
	int nzero = 0;
	for(unsigned i=0; i < ne-1; i++) {
		float red = (float)i/(float)ne;
		
		if(leafHash[i].value >= ne) {
			qDebug()<<"invalid hash value "<<leafHash[i].value;
			nzero++;
		}
		
		ab = boxes[leafHash[i].value];
        
		bb.setMin(ab.low.x, ab.low.y, ab.low.z);
        bb.setMax(ab.high.x, ab.high.y, ab.high.z);
		
		glColor3f(red, 1.f - red, 0.f);
		Aabb a0 = boxes[leafHash[i].value];
		glVertex3f(a0.low.x * 0.5f + a0.high.x * 0.5f, a0.low.y * 0.5f + a0.high.y * 0.5f + 0.2f, a0.low.z * 0.5f + a0.high.z * 0.5f);
        
		Aabb a1 = boxes[leafHash[i+1].value];
		glVertex3f(a1.low.x * 0.5f + a1.high.x * 0.5f, a1.low.y * 0.5f + a1.high.y * 0.5f + 0.2f, a1.low.z * 0.5f + a1.high.z * 0.5f);
        
	}
	glEnd();	
	if(nzero > 0) qDebug()<<"n zero code "<<nzero;
#endif
*/
#endif

}

void GLWidget::clientSelect(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientDeselect(QMouseEvent */*event*/) 
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientMouseInput(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
		case Qt::Key_A:
			m_displayLevel++;
			break;
		case Qt::Key_D:
			m_displayLevel--;
			break;
		case Qt::Key_W:
			internalTimer()->stop();
			break;
		case Qt::Key_S:
			internalTimer()->start();
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}
