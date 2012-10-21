/*
 *  shapeDrawer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#include "shapeDrawer.h"
#include <cmath>

void ShapeDrawer::setGrey(float g)
{
    glColor3f(g, g, g);
}

void ShapeDrawer::setColor(float r, float g, float b)
{
	glColor3f(r, g, b);
}

void ShapeDrawer::box(float width, float height, float depth)
{
	glBegin(GL_LINES);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(width, 0.f, 0.f);
	
	glColor3f(0.f, 1.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, height, 0.f);
	
	glColor3f(0.f, 0.f, 1.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, depth);
	
	glColor3f(0.23f, 0.23f, 0.24f);
	
	glVertex3f(width, 0.f, 0.f);
	glVertex3f(width, 0.f, depth);
	
	glVertex3f(width, 0.f, depth);
	glVertex3f(0.f, 0.f, depth);
	
	glVertex3f(0.f, height, 0.f);
	glVertex3f(width, height, 0.f);
	
	glVertex3f(width, height, 0.f);
	glVertex3f(width, height, depth);
	
	glVertex3f(width, height, depth);
	glVertex3f(0.f, height, depth);
	
	glVertex3f(0.f, height, depth);
	glVertex3f(0.f, height, 0.f);
	
	glVertex3f(width, 0.f, 0.f);
	glVertex3f(width, height, 0.f);
	
	glVertex3f(width, 0.f, depth);
	glVertex3f(width, height, depth);
	
	glVertex3f(0.f, 0.f, depth);
	glVertex3f(0.f, height, depth);
	glEnd();
}

void ShapeDrawer::solidCube(float x, float y, float z, float size)
{
	glBegin(GL_QUADS);
	
// bottom
	glVertex3f(x, y, z);
	glVertex3f(x + size, y, z);
	glVertex3f(x + size, y, z + size);
	glVertex3f(x, y, z + size);

// top
	glVertex3f(x, y+ size, z);
	glVertex3f(x + size, y+ size, z);
	glVertex3f(x + size, y+ size, z + size);
	glVertex3f(x, y+ size, z + size);
	
// back	
	glVertex3f(x, y, z);
	glVertex3f(x, y + size, z);
	glVertex3f(x + size, y + size, z);
	glVertex3f(x + size, y, z);
	
// front	
	glVertex3f(x, y, z + size);
	glVertex3f(x, y + size, z + size);
	glVertex3f(x + size, y + size, z + size);
	glVertex3f(x + size, y, z + size);

// left
	glVertex3f(x, y, z);
	glVertex3f(x, y, z + size);
	glVertex3f(x, y + size, z + size);
	glVertex3f(x, y + size, z);
	
// right
	glVertex3f(x + size, y, z);
	glVertex3f(x + size, y, z + size);
	glVertex3f(x + size, y + size, z + size);
	glVertex3f(x + size, y + size, z);
	glEnd();
}

void ShapeDrawer::end()
{
    if(m_wired) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnd();
}

void ShapeDrawer::beginSolidTriangle()
{
	glBegin(GL_TRIANGLES);
}

void ShapeDrawer::beginWireTriangle()
{
    m_wired = 1;
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_TRIANGLES);
}

void ShapeDrawer::beginLine()
{
	glBegin(GL_LINES);
}

void ShapeDrawer::beginPoint()
{
	glBegin(GL_POINTS);
}

void ShapeDrawer::beginQuad()
{
	glBegin(GL_QUADS);
}

void ShapeDrawer::aVertex(float x, float y, float z)
{
	glVertex3f(x, y, z);
}

void ShapeDrawer::drawVertex(const Polytode * poly)
{
	const int numV = poly->getNumVertex();
	beginPoint();
	for(int i = 0; i < numV; i++) 
	{
		const Vertex p = poly->getVertex(i);
		aVertex(p.x, p.y, p.z);
	}
	end();
}

void ShapeDrawer::drawWiredFace(const Polytode * poly)
{
	beginLine();
	
	const int numFace = poly->getNumFace();
	
	for(int i = 0; i < numFace; i++ )
	{
		const Facet f = poly->getFacet(i);

		Vertex p0 = f.getVertex(0);
		Vertex p1 = f.getVertex(1);
		Vertex p2 = f.getVertex(2);
		
		aVertex(p0.x, p0.y, p0.z);
		aVertex(p1.x, p1.y, p1.z);
		
		aVertex(p1.x, p1.y, p1.z);
		aVertex(p2.x, p2.y, p2.z);
		
		aVertex(p2.x, p2.y, p2.z);
		aVertex(p0.x, p0.y, p0.z);
	}
	
	end();
}

void ShapeDrawer::drawNormal(const Polytode * poly)
{
	const int numFace = poly->getNumFace();
	beginLine();
	for(int i = 0; i < numFace; i++ )
	{
		const Facet f = poly->getFacet(i);

		const Vector3F c = f.getCentroid();
		const Vector3F nor = f.getNormal();
		aVertex(c.x, c.y, c.z);
		aVertex(c.x + nor.x, c.y + nor.y, c.z + nor.z);
	}
	end();
}

void ShapeDrawer::drawSphere()
{
	const float angleDelta = 3.14159269f / 36.f;
	float a0, a1, b0, b1;
	glBegin(GL_LINES);
	for(int i=0; i<72; i++) {
		float angleMin = angleDelta * i;
		float angleMax = angleMin + angleDelta;
		
		a0 = cos(angleMin);
		b0 = sin(angleMin);
		
		a1 = cos(angleMax);
		b1 = sin(angleMax);
		
		glVertex3f(a0, b0, 0.f);
		glVertex3f(a1, b1, 0.f);
		
		glVertex3f(a0, 0.f, b0);
		glVertex3f(a1, 0.f, b1);
		
		glVertex3f(0.f, a0, b0);
		glVertex3f(0.f, a1, b1);
	}
	glEnd();
}

void ShapeDrawer::drawCircleAround(const Vector3F& center)
{
	Vector3F nor(center.x, center.y, center.z);
	Vector3F tangent = nor.perpendicular();
	
	Vector3F v0 = tangent * 0.1f;
	Vector3F p;
	const float delta = 3.14159269f / 9.f;
	
	glBegin(GL_LINES);
	for(int i = 0; i < 18; i++) {
		p = nor + v0;
		glVertex3f(p.x, p.y, p.z);
		
		v0.rotateAroundAxis(nor, delta);
		
		p = nor + v0;
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
}

void ShapeDrawer::drawMesh(const BaseMesh * mesh)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());

// draw a cube
	glDrawElements(GL_TRIANGLES, mesh->getNumFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
}

void ShapeDrawer::drawMesh(const BaseMesh * mesh, const BaseBuffer * buffer)
{
	
	glEnableClientState(GL_VERTEX_ARRAY);
    
	glBindBuffer(GL_ARRAY_BUFFER, buffer->getBufferName());
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glDrawElements(GL_TRIANGLES, mesh->getNumFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
    
}

void ShapeDrawer::drawKdTree(const KdTree * tree)
{
	BoundingBox bbox = tree->m_bbox;
	KdTreeNode * root = tree->getRoot();
	
	setWired(1);
	beginQuad();
	drawKdTreeNode(root, bbox);
	end();
}

void ShapeDrawer::drawKdTreeNode(const KdTreeNode * tree, const BoundingBox & bbox)
{
	Vector3F corner0 = bbox.m_min;
	Vector3F corner1 = bbox.m_max;
	if(tree->isLeaf()) return;

	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	
	glVertex3f(corner0.x, corner1.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	
	int axis = tree->getAxis();
	
	if(axis == 0) {
		corner0.x = corner1.x = tree->getSplitPos();
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner1.z);
		glVertex3f(corner0.x, corner0.y, corner1.z);
	}
	else if(axis == 1) {
		corner0.y = corner1.y = tree->getSplitPos();
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner1.z);
		glVertex3f(corner0.x, corner0.y, corner1.z);
	}
	else {
		corner0.z = corner1.z = tree->getSplitPos();
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner1.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner0.z);
	}
	
	BoundingBox leftBox, rightBox;
	
	float splitPos = tree->getSplitPos();
	bbox.split(axis, splitPos, leftBox, rightBox);
	
	drawKdTreeNode(tree->getLeft(), leftBox);
	drawKdTreeNode(tree->getRight(), rightBox);
	
}

void ShapeDrawer::setWired(char var)
{
	if(var) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

