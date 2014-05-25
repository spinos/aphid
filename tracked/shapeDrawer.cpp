/*
 *  shapeDrawer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#endif

#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#endif

#include "shapeDrawer.h"

class GlDrawcallback : public btTriangleCallback
{
public:
	GlDrawcallback() {}

	virtual void processTriangle(btVector3* triangle,int partId, int triangleIndex)
	{

		(void)triangleIndex;
		(void)partId;
			glColor3f(0, 0, 1);
			
			glBegin(GL_LINES);
			glVertex3d(triangle[0].getX(), triangle[0].getY(), triangle[0].getZ());
			glVertex3d(triangle[1].getX(), triangle[1].getY(), triangle[1].getZ());
			glVertex3d(triangle[2].getX(), triangle[2].getY(), triangle[2].getZ());
			glVertex3d(triangle[1].getX(), triangle[1].getY(), triangle[1].getZ());
			glVertex3d(triangle[2].getX(), triangle[2].getY(), triangle[2].getZ());
			glVertex3d(triangle[0].getX(), triangle[0].getY(), triangle[0].getZ());
			glEnd();
	}
};

inline void glDrawVector(const btVector3& v) { glVertex3d(v[0], v[1], v[2]); }
inline void glDrawVector(const Vector3F& v) { glVertex3f(v.x, v.y, v.z); }
inline void glDrawCoordsys() 
{
    glBegin( GL_LINES );
	glColor3f(1.f, 0.f, 0.f);
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(1.f, 0.f, 0.f); 
	glColor3f(0.f, 1.f, 0.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, 1.f, 0.f); 
	glColor3f(0.f, 0.f, 1.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, 0.f, 1.f);		
	glEnd();
}

void ShapeDrawer::drawGravity(const btVector3 & v)
{
	Vector3F dir(v[0], v[1], v[2]); 
	const float sz = dir.length();
	dir.normalize();
	Matrix44F space;
	space.setFrontOrientation(dir);
	glPushMatrix();
	loadSpace(space);
	glScalef(sz, sz, sz);
	static const float p[42] = { -.2f, 0.f, -2.f,  -.2f, 0.f, -.5f,
								-.2f, 0.f, -.5f, -.4f, 0.f, -.5f,
								-.4f, 0.f, -.5f,  0.f, 0.f,  0.f,
								.2f, 0.f, -2.f, .2f, 0.f, -.5f,
								.2f, 0.f, -.5f, .4f, 0.f, -.5f,
								.4f, 0.f, -.5f,  0.f, 0.f,  0.f,
								-.2f, 0.f, -2.f,  .2f, 0.f,  -2.f};
	glBegin(GL_LINES);
	for(int i=0; i < 14; i++) glVertex3fv(&p[i * 3]);
	glEnd();
	glPopMatrix();
}

void ShapeDrawer::drawConstraint(const btTypedConstraint* constraint)
{
	switch (constraint->getConstraintType()) {
		case HINGE_CONSTRAINT_TYPE:
			drawHingeConstraint(static_cast<const btHingeConstraint*>(constraint));
			break;
		case D6_CONSTRAINT_TYPE:
			drawD6Constraint(static_cast<const btGeneric6DofConstraint*>(constraint));
			break;
		default:
			break;
	}
}

void ShapeDrawer::drawD6Constraint(const btGeneric6DofConstraint* d6f)
{
    const btRigidBody &bodyA =	d6f->getRigidBodyA();
    const btRigidBody &bodyB =	d6f->getRigidBodyB(); 
    
    btTransform transA = d6f->getCalculatedTransformA();
    btTransform transB = d6f->getCalculatedTransformB();
    
    btVector3 angularLower, angularUpper;
    ((btGeneric6DofConstraint*)d6f)->getAngularLowerLimit(angularLower);
    ((btGeneric6DofConstraint*)d6f)->getAngularUpperLimit(angularUpper);
    drawAngularLimit(transA, angularLower, angularUpper);
    
    btTransform tA = bodyA.getWorldTransform();
    btTransform tB = bodyB.getWorldTransform();
    
    btVector3 oriA = transA.getOrigin();
    btVector3 oriB = transB.getOrigin();
    
    btVector3 frmA = tA.getOrigin();
    btVector3 frmB = tB.getOrigin();
    
    glBegin( GL_LINES );
	glColor3f(.7f, 1.f, 0.f);
    glDrawVector(oriA);
    glDrawVector(frmA);
    glDrawVector(oriB);
    glDrawVector(frmB);
	glEnd();
}

void ShapeDrawer::drawObject(const btCollisionObject* object)
{
	if(object->getInternalType() == btCollisionObject::CO_RIGID_BODY) {
        drawRigidBody(btRigidBody::upcast(object));
    }
    else if(object->getInternalType() ==  btCollisionObject::CO_SOFT_BODY) {
        drawSoftBody(btSoftBody::upcast(object));
    }
}

void ShapeDrawer::drawRigidBody(const btRigidBody* body)
{
    glPushMatrix();
    loadWorldSpace(body);
    glDrawCoordsys();
    if(body->getInvMass() < .01f) {
        glColor3f(.1f, .2f, .2f);
    }
    else {   
        if(body->getActivationState() == 1)
            glColor3f(0.f, 1.f, 0.f);
        else
            glColor3f(0.f, 0.f, 1.f);
    }
    const btCollisionShape* shape = body->getCollisionShape();
    drawShape(shape);
    
    glPopMatrix();
}

void ShapeDrawer::drawShape(const btCollisionShape* shape)
{
	switch (shape->getShapeType()) {
		case COMPOUND_SHAPE_PROXYTYPE:
			drawCompound(static_cast<const btCompoundShape*>(shape));
			break;
		case BOX_SHAPE_PROXYTYPE:
			drawBox(static_cast<const btBoxShape*>(shape));
			break;
		case CYLINDER_SHAPE_PROXYTYPE:
			drawCylinder(static_cast<const btCylinderShape*>(shape));
			break;
		case TRIANGLE_MESH_SHAPE_PROXYTYPE:
			drawTriangleMesh(static_cast<const btTriangleMeshShape*>(shape));
			break;
		default:
			break;
	}
}

void ShapeDrawer::drawBox(const btBoxShape * boxShape)
{
	const btVector3 halfExtent = boxShape->getHalfExtentsWithMargin();
	const float sx = halfExtent[0] * 2.f;
	const float sy = halfExtent[1] * 2.f;
	const float sz = halfExtent[2] * 2.f;
	
	glPushMatrix();
	glScalef(sx, sy, sz);
	
	static const float p[24] = { -.5f, -.5f, -.5f,  .5f, -.5f, -.5f,
								.5f,  .5f, -.5f, -.5f,  .5f, -.5f,
								-.5f, -.5f,  .5f,  .5f, -.5f,  .5f,
								.5f,  .5f,  .5f, -.5f,  .5f,  .5f};
					 
	static const int index[24] = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 7, 3, 6, 2, 5, 1, 4, 0};
	
	glBegin(GL_LINES);
	
	for(int i = 0; i < 24; i++)
		glVertex3fv(&p[index[i]  * 3 ]);
	
	glEnd();
	
	glPopMatrix();
}

void ShapeDrawer::drawCylinder(const btCylinderShape * shape)
{
	btVector3 halfExtent = shape->getHalfExtentsWithMargin();
	const float depth = halfExtent[1];
	const float radius = shape->getRadius();
	Vector3F p, q;
	glPushMatrix();
	glScalef(radius, depth, radius);
	glBegin(GL_LINES);

	static const float sins[25] = {0., 0.258819045103, 0.5, 0.707106781187, 0.866025403784, 0.965925826289, 1., 
								0.965925826289, 0.866025403784, 0.707106781187, 0.5, 0.258819045103, 
								0., -0.258819045103, -0.5, -0.707106781187, -0.866025403784, -0.965925826289, -1.,
								-0.965925826289, -0.866025403784, -0.707106781187, -0.5, -0.258819045103, 0. };
	static const float coss[25] = {1., 0.965925826289, 0.866025403784, 0.707106781187, 0.5, 0.258819045103, 0., 
								-0.258819045103, -0.5, -0.707106781187, -0.866025403784, -0.965925826289, -1.,
								-0.965925826289, -0.866025403784, -0.707106781187, -0.5, -0.258819045103, 
								0., 0.258819045103, 0.5, 0.707106781187, 0.866025403784, 0.965925826289, 1.};

	for(int i=0; i < 24; i++) {
		p.x = coss[i];
		p.z = sins[i];
		q.x = coss[i+1];
		q.z = sins[i+1];
		p.y = 1.f;
		q.y = 1.f;
		glDrawVector(p);
		glDrawVector(q);
		p.y = -1.f;
		q.y = -1.f;
		glDrawVector(p);
		glDrawVector(q);
		q = p;
		q.y = 1.f;
		glDrawVector(p);
		glDrawVector(q);
	}
	glEnd();
	glPopMatrix();
}

void ShapeDrawer::drawTriangleMesh(const btTriangleMeshShape * shape)
{
	btVector3 aabbMin, aabbMax;
	btTransform trans; trans.setIdentity();
	shape->getAabb(trans, aabbMin, aabbMax);
	GlDrawcallback drawCallback;
	shape->processAllTriangles(&drawCallback, aabbMin, aabbMax);
}

void ShapeDrawer::drawSoftBody(const btSoftBody* body)
{
    int i;
    glBegin(GL_LINES);
	for(i=0; i < body->m_links.size(); ++i) {
        const btSoftBody::Link&	l = body->m_links[i];
        glDrawVector(l.m_n[0]->m_x);
        glDrawVector(l.m_n[1]->m_x);
    }
	glEnd();
}

void ShapeDrawer::drawTransform(const btRigidBody & body)
{
    glPushMatrix();
    loadWorldSpace(&body);
    glDrawCoordsys();
    glPopMatrix();
}

void ShapeDrawer::loadWorldSpace(const btRigidBody* body)
{
    btScalar m[16];
    
    m[0] = 1.0; m[1] = m[2] = m[3] = 0.0;
    m[4] = 0.0; m[5] = 1.0; m[6] = m[7] = 0.0;
    m[8] = m[9] = 0.0; m[10] = 1.0; m[11] = 0.0;
    m[12] = m[13] = m[14] = 0.0; m[15] = 1.0;
    body->getWorldTransform().getOpenGLMatrix(m);
    glMultMatrixf((const GLfloat*)m);
}

void ShapeDrawer::drawTransform(const btTransform & t)
{
    glPushMatrix();
    btScalar m[16];
    
    m[0] = 1.0; m[1] = m[2] = m[3] = 0.0;
    m[4] = 0.0; m[5] = 1.0; m[6] = m[7] = 0.0;
    m[8] = m[9] = 0.0; m[10] = 1.0; m[11] = 0.0;
    m[12] = m[13] = m[14] = 0.0; m[15] = 1.0;
    t.getOpenGLMatrix(m);
    glMultMatrixf((const GLfloat*)m);
    glDrawCoordsys();
    glPopMatrix();
}

void ShapeDrawer::drawForce(const btRigidBody* body)
{
    btVector3 c = body->getWorldTransform().getOrigin();
    btVector3 f = body->getTotalForce();
    glBegin( GL_LINES );
	glDrawVector(c);
	glDrawVector(c + f);
	glEnd();
}

void ShapeDrawer::drawTranslateHandle(const btRigidBody* body)
{
    btVector3 c = body->getWorldTransform().getOrigin();
    glBegin( GL_LINES );
	
	glColor3f(1.f, 0.f, 0.f);
			glDrawVector(c);
			glDrawVector(c + btVector3(4.f, 0.f, 0.f));
	glColor3f(0.f, 1.f, 0.f);					
			glDrawVector(c);
			glDrawVector(c + btVector3(0.f, 4.f, 0.f)); 
	glColor3f(0.f, 0.f, 1.f);					
			glDrawVector(c);
			glDrawVector(c + btVector3(0.f, 0.f, 4.f)); 	
	glEnd();
}

void ShapeDrawer::drawAngularLimit(const btTransform& space, const btVector3& angularLower, const btVector3& angularUpper)
{
    glPushMatrix();
    btScalar m[16];
    
    m[0] = 1.0; m[1] = m[2] = m[3] = 0.0;
    m[4] = 0.0; m[5] = 1.0; m[6] = m[7] = 0.0;
    m[8] = m[9] = 0.0; m[10] = 1.0; m[11] = 0.0;
    m[12] = m[13] = m[14] = 0.0; m[15] = 1.0;
    space.getOpenGLMatrix(m);
    glMultMatrixf((const GLfloat*)m);
    float x, y, z;
    glBegin( GL_LINES );
	
    if(angularLower.getX() < angularUpper.getX()) {
        glColor3f(1.f, 0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
        
        y = sin((float)angularLower.getX());
        z = cos((float)angularLower.getX());
        glVertex3f(0.f, y, z);
        
        glVertex3f(0.f, 0.f, 0.f);
        
        y = sin((float)angularUpper.getX());
        z = cos((float)angularUpper.getX());
        glVertex3f(0.f, y, z);
    }
	
    if(angularLower.getY() < angularUpper.getY()) {
        glColor3f(0.f, 1.f, 0.f);					
        glVertex3f(0.f, 0.f, 0.f);
        x = cos((float)angularLower.getY());
        z = sin((float)angularLower.getY());
        glVertex3f(x, 0.f, z);
        
        glVertex3f(0.f, 0.f, 0.f);
        x = cos((float)angularUpper.getY());
        z = sin((float)angularUpper.getY());
        glVertex3f(x, 0.f, z);
	}
	
	if(angularLower.getZ() < angularUpper.getZ()) {
        glColor3f(0.f, 0.f, 1.f);					
        glVertex3f(0.f, 0.f, 0.f);
        x = cos((float)angularLower.getZ());
        y = sin((float)angularLower.getZ());	
        glVertex3f(x, y, 0.f);
        
        glVertex3f(0.f, 0.f, 0.f);
        x = cos((float)angularUpper.getZ());
        y = sin((float)angularUpper.getZ());	
        glVertex3f(x, y, 0.f);
	}
	
	glEnd();
    glPopMatrix();
}

void ShapeDrawer::drawHingeConstraint(const btHingeConstraint* constraint)
{
	const btRigidBody &bodyA =	constraint->getRigidBodyA();
    const btRigidBody &bodyB =	constraint->getRigidBodyB(); 
    
    btTransform tA = bodyA.getWorldTransform();
    btTransform tB = bodyB.getWorldTransform();
	
	btTransform frmA = constraint->getAFrame();
	btTransform frmB = constraint->getBFrame();
    
	btVector3 pA = tA.getOrigin();
    btVector3 pB = tB.getOrigin();
	
	btTransform tJ = tA * frmA;
	btVector3 pJ = tJ.getOrigin();
	
	btTransform axis;
	axis.setOrigin(btVector3(0.f, 0.f, 2.f));
	axis = tJ * axis;
	btVector3 pAx = axis.getOrigin();
	
	axis.setOrigin(btVector3(0.f, 0.f, 2.f));
	axis = tB * frmB * axis;
	btVector3 pAx1 = axis.getOrigin();
    
    glBegin( GL_LINES );
	glColor3f(.7f, 1.f, 0.f);
    glDrawVector(pJ);
    glDrawVector(pA);
    glDrawVector(pJ);
	glDrawVector(pB);
	glDrawVector(pJ);
	glDrawVector(pAx);
	glDrawVector(pJ);
	glDrawVector(pAx1);
	glEnd();
}

void ShapeDrawer::drawCompound(const btCompoundShape* shape)
{
	for (int i=0; i<shape->getNumChildShapes(); i++) {
		btTransform childTrans = shape->getChildTransform(i);
		glPushMatrix();
		loadSpace(childTrans);
		
		const btCollisionShape* colShape = shape->getChildShape(i);
		drawShape(colShape);
		
		glPopMatrix();
	}
}

void ShapeDrawer::loadSpace(const btTransform & transform)
{
	btScalar m[16];
    
    m[0] = 1.0; m[1] = m[2] = m[3] = 0.0;
    m[4] = 0.0; m[5] = 1.0; m[6] = m[7] = 0.0;
    m[8] = m[9] = 0.0; m[10] = 1.0; m[11] = 0.0;
    m[12] = m[13] = m[14] = 0.0; m[15] = 1.0;
    transform.getOpenGLMatrix(m);
    glMultMatrixf((const GLfloat*)m);
}

void ShapeDrawer::loadSpace(const Matrix44F & transform)
{
	float m[16];
    
    m[0] = transform.M(0, 0); 
	m[1] = transform.M(0, 1); 
	m[2] = transform.M(0, 2); 
	m[3] = transform.M(0, 3); 
    m[4] = transform.M(1, 0); 
	m[5] = transform.M(1, 1); 
	m[6] = transform.M(1, 2); 
	m[7] = transform.M(1, 3); 
	m[8] = transform.M(2, 0); 
	m[9] = transform.M(2, 1); 
	m[10] = transform.M(2, 2); 
	m[11] = transform.M(2, 3); 
	m[12] = transform.M(3, 0); 
	m[13] = transform.M(3, 1); 
	m[14] = transform.M(3, 2); 
	m[15] = transform.M(3, 3); 
    
	glMultMatrixf((const GLfloat*)m);
}
