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
#include <ogl/GProfile.h>
#include <ogl/GLight.h>

namespace aphid {

class BaseDrawer {
public:
	BaseDrawer ();
	virtual ~BaseDrawer ();
	
	virtual void initializeProfile();

	void setGrey(float g) const;
	void setColor(float r, float g, float b) const;
    void useColor(float * c) const;
	void useColor(const Float3 & c) const;
	void end() const;
	void beginSolidTriangle();
	void beginWireTriangle();
	void beginLine();
	void beginPoint(float x) const;
	void beginQuad() const;
	
	void boundingRectangle(const BoundingRectangle & b) const;
	void boundingBox(const BoundingBox & b) const;
	
	void setWired(char var);
	void setCullFace(char var);
	
	void useSolid() const;
	void useWired() const;
	
	void colorAsActive();
	void colorAsInert();
	void colorAsReference() const;
	void vertex(const Vector3F & v) const;
	void vertexWithOffset(const Vector3F & v, const Vector3F & o);
	void useSpace(const Matrix44F & s) const;
	void useSpace(const Matrix33F & s) const;
	void useDepthTest(char on) const;
	// int loadTexture(int idx, ZEXRImage * image);
	void clearTexture(int idx);
	void texture(int idx);
	void bindTexture(int idx);
	void unbindTexture();
	
	void quad(Vector3F & a, Vector3F & b, Vector3F & c, Vector3F & d, char filled = 0) const;
	
	void setGroupColorLight(int i) const;
	
protected:
	const float * lightColor(int i) const;
	
public:
	GProfile m_markerProfile;
	GProfile m_surfaceProfile;
	GProfile m_wireProfile;
	GProfile m_paintProfile;
	
protected:
	GLight majorLit;
	GLight fillLit;
	GMaterial *surfaceMat;
	GMaterial *paintMat;
	
private:
	int addTexture();
	
private:
	std::vector<GLuint> m_textureNames;
    char m_wired;
	Vector3F m_activeColor, m_inertColor;
};

}