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
#include <BoundingRectangle.h>
#include <SelectionArray.h>
#include <Anchor.h>
#include <GeodesicSphereMesh.h>
#include <PyramidMesh.h>
#include <CubeMesh.h>
#include <BaseCurve.h>
#include <CircleCurve.h>
#include <GProfile.h>
class BaseTransform;
class TransformManipulator;
class SkeletonJoint;
class ZEXRImage;
class LineBuffer;
class BaseDrawer {
public:
	BaseDrawer ();
	virtual ~BaseDrawer ();
	
	virtual void initializeProfile();
	
	void cube(const Vector3F & p, const float & size) const;
	void box(float width, float height, float depth);
	void solidCube(float x, float y, float z, float size);
	void transform(BaseTransform * t);
	void manipulator(TransformManipulator * m);
	void skeletonJoint(SkeletonJoint * joint);
	void moveHandle(int axis, bool active);
	void spinHandle(TransformManipulator * m, bool active);
	void spinPlanes(BaseTransform * t);
	void quadMesh(const BaseMesh * mesh) const;
	
	void setGrey(float g);
	void setColor(float r, float g, float b) const;
	void end() const;
	void beginSolidTriangle();
	void beginWireTriangle();
	void beginLine();
	void beginPoint(float x) const;
	void beginQuad();
	void aVertex(float x, float y, float z);
	
	void boundingRectangle(const BoundingRectangle & b) const;
	void boundingBox(const BoundingBox & b) const;
	
	void drawLineBuffer(LineBuffer * line) const;
	void drawSphere();
	void drawCircleAround(const Vector3F& center);
	void drawMesh(const BaseMesh * mesh, const BaseDeformer * deformer = 0) const;
	void drawPolygons(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void drawPoints(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void showNormal(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void edge(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void field(const BaseField * f);
	void tangentFrame(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void triangle(const BaseMesh * mesh, unsigned idx);
	void patch(const BaseMesh * mesh, unsigned idx);
	void components(SelectionArray * arr);
	void primitive(Primitive * prim);
	void coordsys(float scale = 1.f) const;
	void coordsys(const Matrix33F & orient, float size = 1.f, Vector3F * p = 0) const;
	void setWired(char var);
	void setCullFace(char var);
	void anchor(Anchor *a, char active = 0);
	void spaceHandle(SpaceHandle * hand);
	void sphere(float size = 1.f) const;
	void linearCurve(const BaseCurve & curve);
	void smoothCurve(BaseCurve & curve, short deg);
	void hiddenLine(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void colorAsActive();
	void colorAsInert();
	void vertex(const Vector3F & v) const;
	void vertexWithOffset(const Vector3F & v, const Vector3F & o);
	void circleAt(const Vector3F & pos, const Vector3F & nor);
	void circleAt(const Matrix44F & mat, float radius);
	void arrow0(const Vector3F & at, const Vector3F & dir, float l, float w) const;
	void arrow2(const Vector3F& origin, const Vector3F& dest, float width) const;
	void arrow(const Vector3F& origin, const Vector3F& dest) const;
	void useSpace(const Matrix44F & s) const;
	void useSpace(const Matrix33F & s) const;
	void useDepthTest(char on) const;
	void perVertexVector(BaseMesh * mesh, const std::string & name);
	void vertexNormal(BaseMesh * mesh);
	void lines(const std::vector<Vector3F> & vs);
	void lineStripes(const unsigned & num, unsigned * nv, Vector3F * vs) const;
	void lineStripes(const unsigned & num, unsigned * nv, Vector3F * vs, Vector3F * cs) const;
	
	int loadTexture(int idx, ZEXRImage * image);
	void clearTexture(int idx);
	void texture(int idx);
	void bindTexture(int idx);
	void unbindTexture();
public:
	GProfile m_markerProfile;
	GProfile m_surfaceProfile;
	GProfile m_wireProfile;
	GMaterial *surfaceMat;
	GLight majorLit;
	GLight fillLit;
	
private:
	int addTexture();
	
private:
	std::vector<GLuint> m_textureNames;
    char m_wired;
	GeodesicSphereMesh * m_sphere;
	PyramidMesh * m_pyramid;
	CubeMesh * m_cube;
	CircleCurve * m_circle;
	Vector3F m_activeColor, m_inertColor;
};