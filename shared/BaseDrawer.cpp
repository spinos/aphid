/*
 *  BaseDrawer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif

#include "BaseDrawer.h"
#include "Matrix33F.h"
#include <cmath>

BaseDrawer::BaseDrawer () : m_wired(0) 
{
	m_sphere = new GeodesicSphereMesh(8);
}

BaseDrawer::~BaseDrawer () 
{
	delete m_sphere;
}

void BaseDrawer::setGrey(float g)
{
    glColor3f(g, g, g);
}

void BaseDrawer::setColor(float r, float g, float b)
{
	glColor3f(r, g, b);
}

void BaseDrawer::box(float width, float height, float depth)
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

void BaseDrawer::solidCube(float x, float y, float z, float size)
{
	x -= size * 0.5f;
	y -= size * 0.5f;
	z -= size * 0.5f;
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

void BaseDrawer::end()
{
    if(m_wired) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnd();
}

void BaseDrawer::beginSolidTriangle()
{
	glBegin(GL_TRIANGLES);
}

void BaseDrawer::beginWireTriangle()
{
    m_wired = 1;
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_TRIANGLES);
}

void BaseDrawer::beginLine()
{
	glBegin(GL_LINES);
}

void BaseDrawer::beginPoint()
{
	glBegin(GL_POINTS);
}

void BaseDrawer::beginQuad()
{
	glBegin(GL_QUADS);
}

void BaseDrawer::aVertex(float x, float y, float z)
{
	glVertex3f(x, y, z);
}

void BaseDrawer::drawSphere()
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

void BaseDrawer::drawCircleAround(const Vector3F& center)
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

void BaseDrawer::drawMesh(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedData());

// draw a cube
	glDrawElements(GL_TRIANGLES, mesh->getNumFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::edge(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedData());

// draw a cube
	glDrawElements(GL_LINES, mesh->m_numEdgeVertices, GL_UNSIGNED_INT, mesh->m_edgeIndices);

// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::field(const BaseField * f)
{
	BaseMesh *mesh = f->m_mesh;
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, (GLfloat*)f->getColor());
	
// draw a cube
	glDrawElements(GL_TRIANGLES, mesh->getNumFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::tangentFrame(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	unsigned nv = mesh->getNumVertices();
	Vector3F * v = mesh->getVertices();
	if(deformer)
		v = deformer->getDeformedData();
		
	float m[16];
	for(unsigned i = 0; i < nv; i++) {
		glPushMatrix();
		//glTranslatef(v[i].x, v[i].y, v[i].z);
		Matrix33F orient = mesh->getTangentFrame(i);
    
    m[0] = orient(0, 0); m[1] = orient(0, 1); m[2] = orient(0, 2); m[3] = 0.0;
    m[4] = orient(1, 0); m[5] = orient(1, 1); m[6] = orient(1, 2); m[7] = 0.0;
    m[8] = orient(2, 0); m[9] = orient(2, 1); m[10] = orient(2, 2); m[11] = 0.0;
    m[12] = v[i].x;
	m[13] = v[i].y;
	m[14] = v[i].z; 
	m[15] = 1.0;
    glMultMatrixf((const GLfloat*)m);
		coordsys();
				glPopMatrix();
	}
}

void BaseDrawer::box(const BoundingBox & b)
{
	beginQuad();
	Vector3F corner0(b.getMin(0), b.getMin(1), b.getMin(2));
	Vector3F corner1(b.getMax(0), b.getMax(1), b.getMax(2));

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
	end();
}

void BaseDrawer::triangle(const BaseMesh * mesh, unsigned idx)
{
	beginSolidTriangle();
	Vector3F *v = mesh->getVertices();
	unsigned *i = mesh->getIndices();
	
	Vector3F & a = v[i[idx * 3]];
	Vector3F & b = v[i[idx * 3 + 2]];
	Vector3F & c = v[i[idx * 3 + 1]];
	
	glVertex3f(a.x, a.y, a.z);
	glVertex3f(b.x, b.y, b.z);
	glVertex3f(c.x, c.y, c.z);
	end();
}

void BaseDrawer::components(SelectionArray * arr)
{
    BaseMesh *mesh = (BaseMesh *)arr->getGeometry();
    if(arr->getComponentFilterType() == PrimitiveFilter::TFace) {
        const unsigned numFace = arr->numFaces();
        for(unsigned i = 0; i < numFace; i++) {
            triangle((const BaseMesh *)mesh, arr->getFaceId(i));
        }
    }
    else if(arr->getComponentFilterType() == PrimitiveFilter::TVertex) {
        const unsigned numVert = arr->numVertices();
		if(numVert < 1) return;
		if(numVert < 2) {
			Vector3F * p = arr->getVertexP(0);
            solidCube(p->x, p->y, p->z, 0.2f);
		}
		else {
			BaseCurve curve;
			for(unsigned i = 0; i < numVert; i++) {
				Vector3F * p = arr->getVertexP(i);
				curve.addVertex(*p);
			}
			curve.computeKnots();
			linearCurve(curve);
		}
    }
}

void BaseDrawer::primitive(Primitive * prim)
{
	BaseMesh *mesh = (BaseMesh *)prim->getGeometry();
	unsigned iface = prim->getComponentIndex();
	triangle((const BaseMesh *)mesh, iface);	
}

void BaseDrawer::coordsys(float scale)
{
	glBegin( GL_LINES );
	glColor3f(1.f, 0.f, 0.f);
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(scale, 0.f, 0.f); 
	glColor3f(0.f, 1.f, 0.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, scale, 0.f); 
	glColor3f(0.f, 0.f, 1.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, 0.f, scale);	
	glEnd();
	float arrowSize = scale * 0.1f;
	if(arrowSize < .1f) arrowSize = .1f;
	if(arrowSize > scale * 0.33f) arrowSize = scale * 0.33f;
	const float arrowWidth = arrowSize * .23f;
	setWired(0);
	glBegin(GL_TRIANGLES);
	glColor3f(1.f, 0.f, 0.f);
			glVertex3f(scale - arrowSize, -arrowWidth, arrowWidth);
			glVertex3f(scale - arrowSize,  arrowWidth, arrowWidth);
			glVertex3f(scale - arrowSize,  arrowWidth, -arrowWidth);
			
			glVertex3f(scale - arrowSize,  arrowWidth, -arrowWidth);
			glVertex3f(scale - arrowSize, -arrowWidth, -arrowWidth);
			glVertex3f(scale - arrowSize, -arrowWidth, arrowWidth);
			
			glVertex3f(scale - arrowSize, -arrowWidth, arrowWidth);
			glVertex3f(scale, 0.f, 0.f);
			glVertex3f(scale - arrowSize, arrowWidth, arrowWidth);
			
			glVertex3f(scale - arrowSize, arrowWidth, arrowWidth);
			glVertex3f(scale, 0.f, 0.f);
			glVertex3f(scale - arrowSize, arrowWidth, -arrowWidth);
			
			glVertex3f(scale - arrowSize, arrowWidth, -arrowWidth);
			glVertex3f(scale, 0.f, 0.f);
			glVertex3f(scale - arrowSize, -arrowWidth, -arrowWidth);
			
			glVertex3f(scale - arrowSize, -arrowWidth, -arrowWidth);
			glVertex3f(scale, 0.f, 0.f);
			glVertex3f(scale - arrowSize, -arrowWidth, arrowWidth);
	glColor3f(0.f, 1.f, 0.f);					
			glVertex3f(-arrowWidth, scale - arrowSize, arrowWidth);
			glVertex3f(arrowWidth, scale - arrowSize,  arrowWidth);
			glVertex3f(arrowWidth, scale - arrowSize,  -arrowWidth);
			
			glVertex3f(arrowWidth, scale - arrowSize,  -arrowWidth);
			glVertex3f(-arrowWidth, scale - arrowSize, -arrowWidth);
			glVertex3f(-arrowWidth, scale - arrowSize, arrowWidth);
			
			glVertex3f(arrowWidth, scale - arrowSize, arrowWidth);
			glVertex3f(0.f, scale, 0.f);
			glVertex3f(-arrowWidth, scale - arrowSize, arrowWidth);
			
			glVertex3f(arrowWidth, scale - arrowSize, -arrowWidth);
			glVertex3f(0.f, scale, 0.f);
			glVertex3f(arrowWidth, scale - arrowSize, arrowWidth);
			
			glVertex3f(-arrowWidth, scale - arrowSize, -arrowWidth);
			glVertex3f(0.f, scale, 0.f);
			glVertex3f(arrowWidth, scale - arrowSize, -arrowWidth);
			
			glVertex3f(-arrowWidth, scale - arrowSize, arrowWidth);
			glVertex3f(0.f, scale, 0.f);
			glVertex3f(-arrowWidth, scale - arrowSize, -arrowWidth);
			
	glColor3f(0.f, 0.f, 1.f);					
			glVertex3f(arrowWidth, -arrowWidth, scale - arrowSize);
			glVertex3f(-arrowWidth, -arrowWidth, scale - arrowSize);
			glVertex3f(arrowWidth,  arrowWidth, scale - arrowSize);
			
			glVertex3f(-arrowWidth, -arrowWidth, scale - arrowSize);
			glVertex3f(-arrowWidth, arrowWidth, scale - arrowSize);
			glVertex3f(arrowWidth,  arrowWidth, scale - arrowSize);
			
			glVertex3f(-arrowWidth, arrowWidth, scale - arrowSize);
			glVertex3f(0.f, 0.f, scale);
			glVertex3f(arrowWidth, arrowWidth, scale - arrowSize);
			
			glVertex3f(arrowWidth, arrowWidth, scale - arrowSize);
			glVertex3f(0.f, 0.f, scale);
			glVertex3f(arrowWidth, -arrowWidth, scale - arrowSize);
			
			glVertex3f(arrowWidth, -arrowWidth, scale - arrowSize);
			glVertex3f(0.f, 0.f, scale);
			glVertex3f(-arrowWidth, -arrowWidth, scale - arrowSize);
			
			glVertex3f(-arrowWidth, -arrowWidth, scale - arrowSize);
			glVertex3f(0.f, 0.f, scale);
			glVertex3f(-arrowWidth, arrowWidth, scale - arrowSize);
	glEnd();
	setWired(1);
}

void BaseDrawer::coordsys(const Matrix33F & orient, float scale)
{
	float m[16];
	m[0] = orient(0, 0); m[1] = orient(0, 1); m[2] = orient(0, 2); m[3] = 0.0;
	m[4] = orient(1, 0); m[5] = orient(1, 1); m[6] = orient(1, 2); m[7] = 0.0;
	m[8] = orient(2, 0); m[9] = orient(2, 1); m[10] = orient(2, 2); m[11] = 0.0;
	m[12] = 0.f;
	m[13] = 0.f;
	m[14] = 0.f; 
	m[15] = 1.f;
	glMultMatrixf((const GLfloat*)m);
	coordsys(scale);
}

void BaseDrawer::setWired(char var)
{
	if(var) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void BaseDrawer::anchor(Anchor *a, float size)
{
	glPushMatrix();
	float m[16];
    
    a->spaceMatrix(m);
    
    glMultMatrixf((const GLfloat*)m);
	sphere(size);

	unsigned nouse;
	for(Anchor::AnchorPoint * ap = a->firstPoint(nouse); a->hasPoint(); ap = a->nextPoint(nouse)) {
		Vector3F p = ap->p;
		solidCube(p.x, p.y, p.z, 0.2f);
	}
	if(a->numPoints() > 1) {
		BaseCurve curve;
		for(unsigned i = 0; i < a->numPoints(); i++)
			curve.addVertex(a->getPoint(i)->p);
		curve.computeKnots();
		linearCurve(curve);
	}
	glPopMatrix();
}

void BaseDrawer::sphere(float size)
{
	glPushMatrix();
	glScalef(size, size, size);
	drawMesh(m_sphere);
	glPopMatrix();
}

void BaseDrawer::linearCurve(const BaseCurve & curve)
{
	float t;
	Vector3F p = curve.getVertex(0);
	solidCube(p.x, p.y, p.z, 0.2f);
	p = curve.getVertex(curve.numVertices() - 1);
	solidCube(p.x, p.y, p.z, 0.2f);
	glBegin(GL_LINE_STRIP);
	for(unsigned i = 0; i < curve.numVertices(); i++) {
		p = curve.getVertex(i);
		t = curve.getKnot(i);
		setColor(1.f - t, 0.f, t);
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
}

