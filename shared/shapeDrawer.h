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
#include <BaseMesh.h>
#include <BaseBuffer.h>
#include <KdTree.h>

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
	void beginQuad();
	void aVertex(float x, float y, float z);
	void drawVertex(const Polytode * poly);
	void drawWiredFace(const Polytode * poly);
	void drawNormal(const Polytode * poly);
	void drawSphere();
	void drawCircleAround(const Vector3F& center);
	void drawMesh(const BaseMesh * mesh);
	void drawMesh(const BaseMesh * mesh, const BaseBuffer * buffer);
	void drawKdTree(const KdTree * tree);
	void drawKdTreeNode(const KdTreeNode * tree, const BoundingBox & bbox);
	void setWired(char var);
	
private:
    char m_wired;
};