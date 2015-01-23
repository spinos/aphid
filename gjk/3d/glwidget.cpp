#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <GjkContactSolver.h>
	
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
	
    m_tetrahedron[0].set(-2.5f, -2.5f, -2.5f);
    m_tetrahedron[1].set(-2.5f, -2.5f, 2.5f);
    m_tetrahedron[2].set(2.f, -2.5f, -2.5f);
    m_tetrahedron[3].set(0.f, 2.5f, 0.f);

    m_alpha = 0.f;
    m_drawLevel = 1;
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::testLine()
{
    Matrix44F mat;
    Vector3F line[2];
    mat.rotateZ(m_alpha);
    mat.rotateX(m_alpha);
    mat.translate(10.f, 10.f, 1.f);
    
    line[0] = mat.transform(Vector3F(-3.f, 0.f, 4.f));
    line[1] = mat.transform(Vector3F(3.f, 0.f, -4.f));
	
	ClosestTestContext result;
	result.hasResult = 0;
	result.distance = 1e9;
	result.needContributes = 1;
	result.referencePoint = Vector3F(8.f, 8.f * sin(m_alpha) + 2.f, 4.f * cos(m_alpha) + 4.f);
	
    closestOnLine(line, &result);
    
	glBegin(GL_LINES);
    glColor3f(result.contributes.x, 1.f - result.contributes.x, 0.f);
    glVertex3f(line[0].x , line[0].y, line[0].z);
	glColor3f(result.contributes.y, 1.f - result.contributes.y, 0.f);
    glVertex3f(line[1].x , line[1].y, line[1].z);

	glColor3f(0.f, 1.f, 1.f);
    glVertex3f(result.referencePoint.x, result.referencePoint.y, result.referencePoint.z);
    Vector3F q = result.resultPoint;
    glVertex3f(q.x, q.y, q.z);
    glEnd();
}

void GLWidget::testTriangle()
{
	Matrix44F mat;
    mat.rotateZ(m_alpha);
    mat.rotateX(m_alpha);
    mat.translate(20.f, 10.f, 1.f);
	
	Vector3F tri[3];
    tri[0] = mat.transform(Vector3F(-4.f, 0.f,-4.f));
    tri[1] = mat.transform(Vector3F(4.f, 0.f, -4.f));
	tri[2] = mat.transform(Vector3F(0.f, 0.f, 4.f));
	
	ClosestTestContext result;
	result.hasResult = 0;
	result.distance = 1e9;
	result.needContributes = 1;
	result.referencePoint = Vector3F(20.f + 5.f * sin(-m_alpha), 4.f * cos(m_alpha) + 1.f, 0.f);
	
	closestOnTriangle(tri, &result);
	
	glBegin(GL_LINES);
	glColor3f(0.f, 1.f, 1.f);
	glVertex3f(result.referencePoint.x, result.referencePoint.y, result.referencePoint.z);
    glVertex3f(result.resultPoint.x, result.resultPoint.y, result.resultPoint.z);
	glEnd();
	
	glBegin(GL_LINE_LOOP);
    glColor3f(result.contributes.x, 1.f - result.contributes.x, 0.f);
    glVertex3f(tri[0].x , tri[0].y, tri[0].z);
	glColor3f(result.contributes.y, 1.f - result.contributes.y, 0.f);
    glVertex3f(tri[1].x , tri[1].y, tri[1].z);
	glColor3f(result.contributes.z, 1.f - result.contributes.z, 0.f);
    glVertex3f(tri[2].x , tri[2].y, tri[2].z);
	glEnd();
}

void GLWidget::testTetrahedron()
{
    Matrix44F mat;
    mat.rotateZ(m_alpha);
    mat.rotateX(m_alpha);
    mat.translate(30.f, 10.f, 1.f);
    
    Vector3F q[4];
    for(int i = 0; i < 4; i++)
        q[i] = mat.transform(m_tetrahedron[i]);
		
	ClosestTestContext result;
	result.hasResult = 0;
	result.distance = 1e9;
	result.needContributes = 1;
	result.referencePoint = Vector3F(28.f + 1.f * sin(-m_alpha), 1.f * cos(-m_alpha) + 1.f, 1.f);
	
	closestOnTetrahedron(q, &result);

    glBegin(GL_LINES);
    Vector3F p;

    glColor3f(0.f, 1.f, 1.f);
	glVertex3f(result.referencePoint.x, result.referencePoint.y, result.referencePoint.z);
    glVertex3f(result.resultPoint.x, result.resultPoint.y, result.resultPoint.z);
	
    //glVertex3f(0.f ,0.f, 0.f);
    //p = closestToOriginOnTetrahedron(q);
    //glVertex3f(p.x, p.y, p.z);
	
	Vector3F c[4];
	c[0].set(result.contributes.x, 1.f - result.contributes.x, 0.f);
    c[1].set(result.contributes.y, 1.f - result.contributes.y, 0.f);
    c[2].set(result.contributes.z, 1.f - result.contributes.z, 0.f);
    c[3].set(result.contributes.w, 1.f - result.contributes.w, 0.f);
   
// 0 - 1
	p = c[0];
	glColor3f(p.x, p.y, p.z);
    p = q[0];
    glVertex3f(p.x, p.y, p.z);
	
	p = c[1];
	glColor3f(p.x, p.y, p.z);
    p = q[1];
    glVertex3f(p.x, p.y, p.z);

// 1 - 2 
	p = c[1];
	glColor3f(p.x, p.y, p.z);
    p = q[1];
    glVertex3f(p.x, p.y, p.z);
	
	p = c[2];
	glColor3f(p.x, p.y, p.z);
    p = q[2];
    glVertex3f(p.x, p.y, p.z);
 
// 2 - 0
	glVertex3f(p.x, p.y, p.z);
	p = c[0];
	glColor3f(p.x, p.y, p.z);
    p = q[0];
    glVertex3f(p.x, p.y, p.z);

// 0 - 3
    glVertex3f(p.x, p.y, p.z);
	p = c[3];
	glColor3f(p.x, p.y, p.z);
    p = q[3];
    glVertex3f(p.x, p.y, p.z);
    
// 3 - 1    
    glVertex3f(p.x, p.y, p.z);
	p = c[1];
	glColor3f(p.x, p.y, p.z);
    p = q[1];
    glVertex3f(p.x, p.y, p.z);

// 3- 2
	p = c[3];
	glColor3f(p.x, p.y, p.z);
    p = q[3];
    glVertex3f(p.x, p.y, p.z);
	p = c[2];
	glColor3f(p.x, p.y, p.z);
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
        glBegin(GL_TRIANGLES);
        glVertex3f(s.p[0].x, s.p[0].y, s.p[0].z);
        glVertex3f(s.p[1].x, s.p[1].y, s.p[1].z);
        glVertex3f(s.p[2].x, s.p[2].y, s.p[2].z);
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
    Vector3F pa[3]; 
    pa[0].set(-3.f, -2.f, 0.f);
	pa[1].set(3.f, -2.f, 0.f);
	pa[2].set(0.f, 2.f, 0.f);
	
	Vector3F pb[3];
	pb[0].set(-2.f, -2.f, 0.f);
	pb[1].set(2.f, -2.f, 0.f);
	pb[2].set(3.f, 2.f, 0.f);
	
	Matrix44F mat;
    mat.rotateZ(m_alpha);
    mat.rotateX(m_alpha);
    mat.translate(12.f, 2.f, 3.f);
	for(int i = 0; i < 3; i++)
	    A.X[i] = mat.transform(pa[i]);
	
	mat.setIdentity();
	mat.rotateZ(-m_alpha * .5f);
    mat.rotateY(-m_alpha);
    mat.translate(12.f + 3.f * sin(m_alpha * 2.f), 2.f, 3.f + 1.f * cos(m_alpha * 2.f));
	for(int i = 0; i < 3; i++)
	    B.X[i] = mat.transform(pb[i]);
		
	GjkContactSolver gjk;
	ClosestTestContext result;
	result.referencePoint.setZero();
	result.needContributes = 1;
	result.distance = 1e9;
	result.hasResult = 0;
	
	resetSimplex(result.W);
	
	gjk.distance(A, B, &result);
	
	glBegin(GL_TRIANGLES);
	
	float grey = 0.f;
	if(result.hasResult) grey = .3f;
    
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
	
	if(result.hasResult) drawSimplex(result.W);
	
	glColor3f(1.f, 1.f, 1.f);
	glBegin(GL_LINES);
	q = Vector3F::Zero;
	glVertex3f(q.x, q.y, q.z);
	q -= result.resultPoint;
	glVertex3f(q.x, q.y, q.z);
	glEnd();
	
}

void GLWidget::testShapeCast()
{
	Vector3F pa[3]; 
    pa[0].set(-1.5f, -3.f, 0.f);
	pa[1].set(2.f, -2.5f, 0.1f);
	pa[2].set(-2.f, 2.5f, 0.f);
	
	Vector3F pb[3];
	pb[0].set(-2.f, 2.f, 0.f);
	pb[1].set(2.f, 2.f, 0.f);
	pb[2].set(2.f, -2.f, 0.f);
	
	Matrix44F mat;
    mat.rotateZ(sin(m_alpha)* 3.f);
	mat.rotateX(-m_alpha);
    
	mat.translate(1.f, 1.f, -1.f);
	for(int i = 0; i < 3; i++)
	    A.X[i] = mat.transform(pa[i]);
		
	mat.setIdentity();
	mat.rotateZ(-m_alpha * 1.5f);
	mat.rotateY(0.2 * sin(m_alpha));
     
    mat.translate(1.f, 1.f, 8.f);
	for(int i = 0; i < 3; i++)
	    B.X[i] = mat.transform(pb[i]);
		
	Vector3F rayDir(.99f * sin(m_alpha * 2.f), .2f * cos(m_alpha), -1.4f);
	rayDir.normalize();
	Vector3F rayBegin= B.X[0];
	Vector3F rayEnd = rayBegin + rayDir * 16.f;
	getDrawer()->arrow(rayBegin, rayEnd);
	
	rayBegin= B.X[1];
	rayEnd = rayBegin + rayDir * 16.f;
	getDrawer()->arrow(rayBegin, rayEnd);
	
	rayBegin= B.X[2];
	rayEnd = rayBegin + rayDir * 16.f;
	getDrawer()->arrow(rayBegin, rayEnd);
		
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_TRIANGLES);
	
    Vector3F q;
    
    glColor3f(1.f, 0.f, 0.f);
    q = A.X[0];
    glVertex3f(q.x, q.y, q.z);
    q = A.X[1];
    glVertex3f(q.x, q.y, q.z);
    q = A.X[2];
    glVertex3f(q.x, q.y, q.z);
    
    glColor3f(0.f, 1.f, 0.f);
    q = B.X[0];
    glVertex3f(q.x, q.y, q.z);
    q = B.X[1];
    glVertex3f(q.x, q.y, q.z);
    q = B.X[2];
    glVertex3f(q.x, q.y, q.z);
    
    glEnd();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	GjkContactSolver gjk;
	
	ClosestTestContext result;
	result.rayDirection = rayDir;
	result.referencePoint.setZero();
	result.needContributes = 1;
	result.distance = 1e9;
	result.hasResult = 0;
	
	resetSimplex(result.W);
	
	gjk.rayCast(A, B, &result);
	
	if(!result.hasResult) return;
	
	glBegin(GL_TRIANGLES);
    
    glColor3f(1.f, 0.f, 0.f);
    q = A.X[0];
    glVertex3f(q.x, q.y, q.z);
    q = A.X[1];
    glVertex3f(q.x, q.y, q.z);
    q = A.X[2];
    glVertex3f(q.x, q.y, q.z);
	
	glColor3f(0.f, 1.f, 0.f);
    q = B.X[0] + result.rayDirection * result.distance;
    glVertex3f(q.x, q.y, q.z);
    q = B.X[1] + result.rayDirection * result.distance;
    glVertex3f(q.x, q.y, q.z);
    q = B.X[2] + result.rayDirection * result.distance;
    glVertex3f(q.x, q.y, q.z);
    
    glEnd();
	
	rayBegin= B.X[0];
	rayEnd = rayBegin + rayDir * result.distance;
	getDrawer()->arrow(rayBegin, rayEnd);
	
	rayBegin= B.X[1];
	rayEnd = rayBegin + rayDir * result.distance;
	getDrawer()->arrow(rayBegin, rayEnd);
	
	rayBegin= B.X[2];
	rayEnd = rayBegin + rayDir * result.distance;
	getDrawer()->arrow(rayBegin, rayEnd);
	
	getDrawer()->arrow(Vector3F::Zero, result.resultPoint);
	
}

void GLWidget::clientDraw()
{
	testShapeCast();
	testGjk();
    testLine();
	testTriangle();
    testTetrahedron();
    m_alpha += 0.01f;
}

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
    switch (e->key()) {
		case Qt::Key_A:
			m_drawLevel++;
			break;
		case Qt::Key_D:
			m_drawLevel--;
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

