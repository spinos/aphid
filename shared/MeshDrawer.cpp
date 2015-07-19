/*
 *  MeshDrawer.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "MeshDrawer.h"
#include <BaseMesh.h>
#include <BaseDeformer.h>
#include <BaseField.h>
#include <ATriangleMesh.h>
MeshDrawer::MeshDrawer() {}
MeshDrawer::~MeshDrawer() {}

void MeshDrawer::triangleMesh(const ATriangleMesh * mesh, const BaseDeformer * deformer) const
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->points());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedP());

	glDrawElements(GL_TRIANGLES, mesh->numIndices(), GL_UNSIGNED_INT, mesh->indices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void MeshDrawer::quadMesh(const BaseMesh * mesh) const
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	
	glDrawElements(GL_QUADS, mesh->getNumPolygonFaceVertices(), GL_UNSIGNED_INT, mesh->getPolygonIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void MeshDrawer::drawMesh(const BaseMesh * mesh, const BaseDeformer * deformer) const
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedP());

	glDrawElements(GL_TRIANGLES, mesh->getNumTriangleFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
}

void MeshDrawer::drawPolygons(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	Vector3F * p = mesh->getVertices();
	if(deformer) p = deformer->getDeformedP();
	
	const unsigned nf = mesh->getNumPolygons();
	unsigned fi = 0;
	unsigned *fc = mesh->getPolygonCounts();
	unsigned *fv = mesh->getPolygonIndices();
	for(unsigned i = 0; i < nf; i++) {
		glBegin(GL_POLYGON);
		for(unsigned j =0; j < fc[i]; j++) {
			vertex(p[fv[fi + j]]);
		}
		glEnd();
		fi += fc[i];
	}
}

void MeshDrawer::drawPoints(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedP());

	glDrawArrays(GL_POINTS, 0, mesh->getNumVertices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void MeshDrawer::showNormal(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedP());

	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getNormals());
	
	glDrawElements(GL_TRIANGLES, mesh->getNumTriangleFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void MeshDrawer::edge(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	if(mesh->numQuads() < 1) return;

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedP());

// draw a cube
	glDrawElements(GL_QUADS, mesh->numQuads() * 4, GL_UNSIGNED_INT, mesh->getQuadIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
}

void MeshDrawer::hiddenLine(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	setCullFace(1);
	setWired(0);
	setGrey(0.4f);
	drawMesh(mesh, deformer);
	setWired(1);
	setGrey(0.9f);
	edge(mesh, deformer);
	setCullFace(0);
}

void MeshDrawer::perVertexVector(BaseMesh * mesh, const std::string & name)
{
	Vector3F *pvv = mesh->perVertexVector(name);
	if(!pvv) return;
	Vector3F * p = mesh->getVertices();
	Vector3F q;
	const unsigned nv = mesh->getNumVertices();
	glBegin(GL_LINES);
	for(unsigned i = 0; i < nv; i++) {
		q = p[i];
		glVertex3fv((float *)&q);
		q += pvv[i];
		glVertex3fv((float *)&q);
	}
	glEnd();
}

void MeshDrawer::triangle(const BaseMesh * mesh, unsigned idx)
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

void MeshDrawer::patch(const BaseMesh * mesh, unsigned idx)
{
	beginQuad();
	Vector3F *v = mesh->getVertices();
	unsigned *i = mesh->getQuadIndices();
	
	Vector3F & a = v[i[idx * 4]];
	Vector3F & b = v[i[idx * 4 + 1]];
	Vector3F & c = v[i[idx * 4 + 2]];
	Vector3F & d = v[i[idx * 4 + 3]];
	
	glVertex3f(a.x, a.y, a.z);
	glVertex3f(b.x, b.y, b.z);
	glVertex3f(c.x, c.y, c.z);
	glVertex3f(d.x, d.y, d.z);
	end();
}

void MeshDrawer::patch(const BaseMesh * mesh, const std::deque<unsigned> & sel) const
{
	if(sel.size() < 1) return;
	if(!mesh) return;
	beginQuad();
	Vector3F *v = mesh->getVertices();
	unsigned *i = mesh->getQuadIndices();
	unsigned j;
	std::deque<unsigned>::const_iterator it = sel.begin();
	for(; it != sel.end(); ++it) {
		j = *it;
		Vector3F & a = v[i[j * 4]];
		Vector3F & b = v[i[j * 4 + 1]];
		Vector3F & c = v[i[j * 4 + 2]];
		Vector3F & d = v[i[j * 4 + 3]];
		
		glVertex3f(a.x, a.y, a.z);
		glVertex3f(b.x, b.y, b.z);
		glVertex3f(c.x, c.y, c.z);
		glVertex3f(d.x, d.y, d.z);
	}
	end();
}

void MeshDrawer::vertexNormal(BaseMesh * mesh)
{
	Vector3F *nor = mesh->getNormals();
	Vector3F * p = mesh->getVertices();
	const unsigned nv = mesh->getNumVertices();
	Vector3F q;
	glBegin(GL_LINES);
	for(unsigned i = 0; i < nv; i++) {
		q = p[i];
		glVertex3fv((float *)&q);
		q += nor[i];
		glVertex3fv((float *)&q);
	}
	glEnd();
}

void MeshDrawer::tangentFrame(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	unsigned nv = mesh->getNumVertices();
	Vector3F * v = mesh->getVertices();
	if(deformer)
		v = deformer->getDeformedP();
		
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
		glColor3f(1.f, 0.f, 0.f);
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(1.f, 0.f, 0.f);
		glColor3f(0.f, 1.f, 0.f);
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(0.f, 1.f, 0.f);
		glColor3f(0.f, 0.f, 1.f);
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(0.f, 0.f, 1.f);
				glPopMatrix();
	}
}

void MeshDrawer::field(const BaseField * f)
{
	BaseMesh *mesh = f->m_mesh;
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, (GLfloat*)f->getColor());
	
// draw a cube
	glDrawElements(GL_TRIANGLES, mesh->getNumTriangleFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}