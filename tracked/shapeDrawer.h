/*
 *  shapeDrawer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <btBulletCollisionCommon.h>
#include <btBulletDynamicsCommon.h>
#include <BulletSoftBody/btSoftBody.h>
#include <AllMath.h>
class ShapeDrawer {
public:
	ShapeDrawer () {}
	virtual ~ShapeDrawer () {}
	
	void box(const float & x, const float & y, const float & z);
	void box(const Matrix44F & mat, const float & x, const float & y, const float & z);
	void cylinder(const float & x, const float & y, const float & z);
	void cylinder(const Matrix44F & mat, const float & x, const float & y, const float & z);
	
	void drawGravity(const btVector3 & v);
	void drawConstraint(const btTypedConstraint* constraint);
	void drawObject(const btCollisionObject* object);
	void drawRigidBody(const btRigidBody* body);
	void drawSoftBody(const btSoftBody* body);
	void drawShape(const btCollisionShape* shape);
	void drawTransform(const btRigidBody & body);
	void loadWorldSpace(const btRigidBody* body);
	void drawTransform(const btTransform & t);
	void drawForce(const btRigidBody* body);
	void drawTranslateHandle(const btRigidBody* body);
	
	void drawCoordsys(const Matrix44F & transform);
private:
	void drawAngularLimit(const btTransform& space, const btTransform& space1, const btVector3& angularLower, const btVector3& angularUpper);
	void drawHingeConstraint(const btHingeConstraint* constraint);
	void drawD6Constraint(const btGeneric6DofConstraint* constraint);
	void drawBox(const btBoxShape * boxShape);
	void drawCylinder(const btCylinderShape * shape);
	void drawTriangleMesh(const btTriangleMeshShape * shape);
	void drawCompound(const btCompoundShape* shape);
	void loadSpace(const btTransform & transform);
	void loadSpace(const Matrix44F & transform);
	void drawCoordsys(const btTransform & transform);
private:

};