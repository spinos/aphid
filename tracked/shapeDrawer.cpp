/*
 *  shapeDrawer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <QGLWidget>
#include <AllMath.h>
#include "shapeDrawer.h"

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
    if(object->getInternalType() ==  btCollisionObject::CO_RIGID_BODY) {
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
		default:
			break;
	}
}

void ShapeDrawer::drawBox(btBoxShape * boxShape)
{
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

void ShapeDrawer::drawCylinder(btCylinderShape * shape)
{
	btVector3 halfExtent = shape->getHalfExtentsWithMargin();
	const float depth = halfExtent[1];
	const float radius = shape->getRadius();
	Vector3F p, q;
	glBegin(GL_LINES);
	int i;
	const float delta = PI / 9.f;
	float alpha = 0.f;
	for(i=0; i < 18; i++) {
		p.x = cos(alpha) * radius;
		p.z = sin(alpha) * radius;
		q.x = cos(alpha + delta) * radius;
		q.z = sin(alpha + delta) * radius;
		p.y = depth;
		q.y = depth;
		glDrawVector(p);
		glDrawVector(q);
		p.y = -depth;
		q.y = -depth;
		glDrawVector(p);
		glDrawVector(q);
		q = p;
		q.y = depth;
		glDrawVector(p);
		glDrawVector(q);
		alpha += delta;
	}
	glEnd();
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

void ShapeDrawer::drawCompound(btCompoundShape* shape)
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
