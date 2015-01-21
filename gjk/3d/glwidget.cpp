#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
	
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
	
    m_tetrahedron[0].set(-1.f, -1.f, -2.f);
    m_tetrahedron[1].set(-.5f, -1.f, 2.f);
    m_tetrahedron[2].set(2.f, -1.f, -1.f);
    m_tetrahedron[3].set(0.f, 2.f, 0.f);

    m_alpha = 0.f;
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::testTetrahedron()
{
    Matrix44F mat;
    // mat.rotateZ(m_alpha);
    mat.rotateX(m_alpha);
    mat.translate(1.f, 1.f, 10.f);
    
    Vector3F q[4];
    for(int i = 0; i < 4; i++)
        q[i] = mat.transform(m_tetrahedron[i]);
    
    glBegin(GL_LINES);
    Vector3F p;
    int nouse;
    glColor3f(0.f, 0.f ,0.5f);
// to closest point from origin
    glVertex3f(0.f ,0.f, 0.f);
    p = closestToOriginOnTetrahedron(q, nouse);
    glVertex3f(p.x, p.y, p.z);
   
    glColor3f(0.f, 0.5f ,0.f);
// 0 - 1
    p = q[0];
    glVertex3f(p.x, p.y, p.z);
    p = q[1];
    glVertex3f(p.x, p.y, p.z);

// 1 - 2 
    p = q[1];
    glVertex3f(p.x, p.y, p.z);
    p = q[2];
    glVertex3f(p.x, p.y, p.z);
 
// 2 - 0
    glVertex3f(p.x, p.y, p.z);
    p = q[0];
    glVertex3f(p.x, p.y, p.z);

// 0 - 3
    glVertex3f(p.x, p.y, p.z);
    p = q[3];
    glVertex3f(p.x, p.y, p.z);
    
// 3 - 1    
    glVertex3f(p.x, p.y, p.z);
    p = q[1];
    glVertex3f(p.x, p.y, p.z);

// 3- 2    
    p = q[3];
    glVertex3f(p.x, p.y, p.z);
    p = q[2];
    glVertex3f(p.x, p.y, p.z);
  
    glEnd();
}

void drawSimplex(const Simplex & s)
{
    if(s.d == 0) return;
    if(s.d == 1) {
        glColor3f(0.f, 1.f, 0.f);
        glBegin(GL_POINTS);
        glVertex3f(s.p[0].x, s.p[0].y, s.p[0].z);
        glEnd();
        return;
    }
    if(s.d == 2) {
        glColor3f(1.f, 1.f, 0.f);
        glBegin(GL_LINES);
        glVertex3f(s.p[0].x, s.p[0].y, s.p[0].z);
        glVertex3f(s.p[1].x, s.p[1].y, s.p[1].z);
        glEnd();
        return;
    }
    if(s.d == 3) {
        glColor3f(0.f, 1.f, 1.f);
        glBegin(GL_LINES);
        glVertex3f(s.p[0].x, s.p[0].y, s.p[0].z);
        glVertex3f(s.p[1].x, s.p[1].y, s.p[1].z);
        glVertex3f(s.p[1].x, s.p[1].y, s.p[1].z);
        glVertex3f(s.p[2].x, s.p[2].y, s.p[2].z);
        glVertex3f(s.p[2].x, s.p[2].y, s.p[2].z);
        glVertex3f(s.p[0].x, s.p[0].y, s.p[0].z);
        glEnd();
        return;
    }
    if(s.d == 4) {
        glColor3f(1.f, 0.f, 1.f);
        glBegin(GL_LINES);
        glVertex3f(s.p[0].x, s.p[0].y, s.p[0].z);
        glVertex3f(s.p[1].x, s.p[1].y, s.p[1].z);
        glVertex3f(s.p[1].x, s.p[1].y, s.p[1].z);
        glVertex3f(s.p[2].x, s.p[2].y, s.p[2].z);
        glVertex3f(s.p[2].x, s.p[2].y, s.p[2].z);
        glVertex3f(s.p[0].x, s.p[0].y, s.p[0].z);
        glVertex3f(s.p[0].x, s.p[0].y, s.p[0].z);
        glVertex3f(s.p[3].x, s.p[3].y, s.p[3].z);
        glVertex3f(s.p[1].x, s.p[1].y, s.p[1].z);
        glVertex3f(s.p[3].x, s.p[3].y, s.p[3].z);
        glVertex3f(s.p[2].x, s.p[2].y, s.p[2].z);
        glVertex3f(s.p[3].x, s.p[3].y, s.p[3].z);
        glEnd();
        return;
    }
}

void GLWidget::testGjk()
{
    Matrix44F mat;

    Vector3F pa[3]; 
    pa[0].set(-1.f, 0.f, -1.f);
	pa[1].set(0.f, 0.f, 3.f);
	pa[2].set(3.f, 0.f, 0.f);
	
	Vector3F pb[3];
	pb[0].set(2.f, -1.1f, -.5f);
	pb[1].set(3.f, 2.f, 0.5f);
	pb[2].set(-1.3f, 0.f, 0.f);
	
	mat.rotateZ(m_alpha);
    mat.rotateX(m_alpha);
    mat.translate(2.f, 2.f, 2.f);
	for(int i = 0; i < 3; i++)
	    A.X[i] = mat.transform(pa[i]);
	
	mat.setIdentity();
	mat.rotateY(m_alpha);
    mat.rotateX(m_alpha);
    mat.translate(2.f, 2.f + 4.f * sin(m_alpha * 2.f), 2.f);
	for(int i = 0; i < 3; i++)
	    B.X[i] = mat.transform(pb[i]);
    
    int k = 0;
	Vector3F w;
	Vector3F v = A.X[0] - B.X[0];
	
	char contacted = 0;
	Simplex W;
	for(int i=0; i < 99; i++) {
	    v.reverse();
	    w = A.supportPoint(v) - B.supportPoint(v.reversed());
	    
	    // std::cout<<" v"<<k<<" "<<v.str()<<"\n";	
	    // std::cout<<" w"<<k<<" "<<w.str()<<"\n";	
	    // std::cout<<" wTv "<<w.dot(v)<<"\n";
	    
	    if(w.dot(v) < 0.f) {
	        std::cout<<" minkowski difference contains the origin\n";
	        std::cout<<"separating axis ||v"<<k<<"|| "<<v.length()<<"\n";
	        break;
	    }
	    
	    addToSimplex(W, w);
	    
	    drawSimplex(W);
	    
	    if(isOriginInsideSimplex(W)) {
	        std::cout<<" simplex W"<<k<<" contains origin, intersected\n";
	        contacted = 1;
	        break;
	    }
	    
	    // std::cout<<" W"<<k<<" d="<<W.d<<"\n";
	    
	    v = closestToOriginWithinSimplex(W);
	    
	    glColor3f(1.f, 0.f, 0.f);
	    glBegin(GL_LINES);
	    glVertex3f(0.f, 0.f, 0.f);
	    glVertex3f(v.x, v.y, v.z);
	    glEnd();
	    
	    k++;
	}
	
	glBegin(GL_TRIANGLES);
	
	float grey = 0.f;
	if(contacted) grey = .3f;
    
    Vector3F q;
    
    glColor3f(0.5f + grey, 0.5f ,0.f);
    q = A.X[0];
    glVertex3f(q.x, q.y, q.z);
    q = A.X[1];
    glVertex3f(q.x, q.y, q.z);
    q = A.X[2];
    glVertex3f(q.x, q.y, q.z);
    
    glColor3f(0.f, 0.5f + grey ,0.5f);
    q = B.X[0];
    glVertex3f(q.x, q.y, q.z);
    q = B.X[1];
    glVertex3f(q.x, q.y, q.z);
    q = B.X[2];
    glVertex3f(q.x, q.y, q.z);
    
    glEnd();
}

void GLWidget::clientDraw()
{
    testGjk();
    testTetrahedron();
    m_alpha += 0.02f;
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void GLWidget::clientDeselect()
{
}

//! [10]
void GLWidget::clientMouseInput(Vector3F & stir)
{
}
//! [10]

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

