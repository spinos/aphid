/*
 *  BaseDrawer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <AllMath.h>
#include <BaseMesh.h>
#include <BaseDeformer.h>
#include <BaseField.h>
#include <BoundingBox.h>
#include <SelectionArray.h>
#include <Anchor.h>
#include <GeodesicSphereMesh.h>
#include <PyramidMesh.h>
#include <CubeMesh.h>
#include <BaseCurve.h>

class BaseDrawer {
public:
	BaseDrawer ();
	virtual ~BaseDrawer ();
	
	void box(float width, float height, float depth);
	void solidCube(float x, float y, float z, float size);
	void setGrey(float g);
	void setColor(float r, float g, float b);
	void end();
	void beginSolidTriangle();
	void beginWireTriangle();
	void beginLine();
	void beginPoint();
	void beginQuad();
	void aVertex(float x, float y, float z);
	void drawSphere();
	void drawCircleAround(const Vector3F& center);
	void drawMesh(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void showNormal(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void edge(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void field(const BaseField * f);
	void tangentFrame(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void box(const BoundingBox & b);
	void triangle(const BaseMesh * mesh, unsigned idx);
	void components(SelectionArray * arr);
	void primitive(Primitive * prim);
	void coordsys(float scale = 1.f);
	void coordsys(const Matrix33F & orient, float size = 1.f);
	void setWired(char var);
	void setCullFace(char var);
	void anchor(Anchor *a, char active = 0);
	void spaceHandle(SpaceHandle * hand);
	void sphere(float size = 1.f);
	void linearCurve(const BaseCurve & curve);
	void hiddenLine(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void colorAsActive();
	void colorAsInert();
	void vertexWithOffset(const Vector3F & v, const Vector3F & o);
	
private:
    char m_wired;
	GeodesicSphereMesh * m_sphere;
	PyramidMesh * m_pyramid;
	CubeMesh * m_cube;
	Vector3F m_activeColor, m_inertColor;
};