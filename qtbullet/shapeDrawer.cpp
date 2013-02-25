/*
 *  shapeDrawer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <QGLWidget>
#include "shapeDrawer.h"

inline void glDrawVector(const btVector3& v) { glVertex3d(v[0], v[1], v[2]); }
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

void ShapeDrawer::drawConstraint(const btTypedConstraint* constraint)
{
    const btRigidBody &bodyA =	constraint->getRigidBodyA();
    const btRigidBody &bodyB =	constraint->getRigidBodyB(); 
    
    const btGeneric6DofConstraint* d6f = static_cast<const btGeneric6DofConstraint*>(constraint);
    btTransform transA = d6f->getCalculatedTransformA();
    btTransform transB = d6f->getCalculatedTransformB();
    drawTransform(transA);
    drawTransform(transB);
    
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
    const btRigidBody* body = btRigidBody::upcast(object);
    glPushMatrix();
    loadWorldSpace(body);
    glDrawCoordsys();
    if(object->getActivationState() == 1)
        glColor3f(0.f, 1.f, 0.f);
    else
        glColor3f(0.f, 0.f, 1.f);
    const btCollisionShape* shape = object->getCollisionShape();
    drawShape(shape);
    
    glPopMatrix();
}

void ShapeDrawer::drawShape(const btCollisionShape* shape)
{
	const btBoxShape* boxShape = static_cast<const btBoxShape*>(shape);
	btVector3 halfExtent = boxShape->getHalfExtentsWithMargin();
	
	btVector3 org(0,0,0);
	btVector3 dx(1,0,0);
	btVector3 dy(0,1,0);
	btVector3 dz(0,0,1);
	dx *= halfExtent[0];
	dy *= halfExtent[1];
	dz *= halfExtent[2];
	
	
	glBegin(GL_LINE_LOOP);
	glDrawVector(org - dx - dy - dz);
	glDrawVector(org + dx - dy - dz);
	glDrawVector(org + dx + dy - dz);
	glDrawVector(org - dx + dy - dz);
	glDrawVector(org - dx + dy + dz);
	glDrawVector(org + dx + dy + dz);
	glDrawVector(org + dx - dy + dz);
	glDrawVector(org - dx - dy + dz);
	glEnd();
	glBegin(GL_LINES);
	glDrawVector(org + dx - dy - dz);
	glDrawVector(org + dx - dy + dz);
	glDrawVector(org + dx + dy - dz);
	glDrawVector(org + dx + dy + dz);
	glDrawVector(org - dx - dy - dz);
	glDrawVector(org - dx + dy - dz);
	glDrawVector(org - dx - dy + dz);
	glDrawVector(org - dx + dy + dz);
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
