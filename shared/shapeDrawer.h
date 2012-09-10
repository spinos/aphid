/*
 *  shapeDrawer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <Polytode.h>

class ShapeDrawer {
public:
	ShapeDrawer () : m_wired(0) {}
	virtual ~ShapeDrawer () {}
	
	void box(float width, float height, float depth);
	void solidCube(float x, float y, float z, float size);
	void setGrey(float g);
	void setColor(float r, float g, float b);
	void end();
	void beginSolidTriangle();
	void beginWireTriangle();
	void beginLine();
	void beginPoint();
	void aVertex(float x, float y, float z);
	void drawVertex(const Polytode * poly);
	void drawWiredFace(const Polytode * poly);
	void drawNormal(const Polytode * poly);
	
private:
    char m_wired;
};