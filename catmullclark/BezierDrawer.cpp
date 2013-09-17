/*
 *  BezierDrawer.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 6/4/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif
#include "tessellator.h"
#include "BezierDrawer.h"
#include <accPatch.h>

BezierDrawer::BezierDrawer() : m_vertices(0), m_normals(0), m_indices(0)
{
	m_tess = new Tessellator;
}

BezierDrawer::~BezierDrawer()
{
	cleanup();
	delete m_tess;
}

void BezierDrawer::updateMesh(AccPatchMesh * mesh)
{
	m_mesh = mesh;
	AccPatch* bez = mesh->beziers();
	cleanup();
	
	m_tess->setNumSeg(4);
	const unsigned numFace = mesh->getNumFaces();
	const unsigned vpf = m_tess->numVertices();
	const unsigned ipf = m_tess->numIndices();
	m_vertices = new Vector3F[vpf * numFace];
	m_normals = new Vector3F[vpf * numFace];
	m_indices = new unsigned[ipf * numFace];
	
	unsigned curP = 0, curI = 0, faceStart;
	unsigned i, j;
	for(i = 0; i < numFace; i++) {
		m_tess->evaluate(bez[i]);
		Vector3F *pop = m_tess->_positions;
		Vector3F *nor = m_tess->_normals;
		int *idr = m_tess->getVertices();
		for(j = 0; j < vpf; j++) {
			m_vertices[curP] = pop[j];
			m_normals[curP] = nor[j];
			curP++;
		}
		faceStart = vpf * i;
		for(j = 0; j < ipf; j++) {
			m_indices[curI] = faceStart + idr[j];
			curI++;
		}
	}
}

void BezierDrawer::drawBezierPatch(BezierPatch * patch)
{
	int seg = 4;
	m_tess->setNumSeg(seg);
	m_tess->evaluate(*patch);
	glColor3f(0.f, 0.3f, 0.9f);
	glEnable(GL_CULL_FACE);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 0, m_tess->getPositions() );
	
	glEnableClientState( GL_COLOR_ARRAY );
	glColorPointer( 3, GL_FLOAT, 0, m_tess->getNormals() );

	glDrawElements( GL_QUADS, seg * seg * 4, GL_UNSIGNED_INT, m_tess->getVertices() );
	glDisableClientState( GL_COLOR_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );
}

void BezierDrawer::drawBezierCage(BezierPatch * patch)
{
	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_QUADS);
	//glColor3f(1,1,1);
	Vector3F p;
	for(unsigned j=0; j < 3; j++) {
		for(unsigned i = 0; i < 3; i++) {
			p = patch->p(i, j);
			glVertex3f(p.x, p.y, p.z);
			p = patch->p(i + 1, j);
			glVertex3f(p.x, p.y, p.z);
			p = patch->p(i + 1, j + 1);
			glVertex3f(p.x, p.y, p.z);
			p = patch->p(i, j + 1);
			glVertex3f(p.x, p.y, p.z);
		}
	}
	glEnd();
}

void BezierDrawer::drawAcc() const
{
	glEnable(GL_CULL_FACE);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 0, m_vertices );
	
	glEnableClientState( GL_COLOR_ARRAY );
	glColorPointer( 3, GL_FLOAT, 0, m_normals );

	glDrawElements( GL_QUADS, m_tess->numIndices() * m_mesh->getNumFaces(), GL_UNSIGNED_INT, m_indices);
	glDisableClientState( GL_COLOR_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );
}

void BezierDrawer::cleanup()
{
	if(m_vertices) delete[] m_vertices;
	if(m_normals) delete[] m_normals;
	if(m_indices) delete[] m_indices;
}

void BezierDrawer::verbose() const
{
	const unsigned numFace = m_mesh->getNumFaces();
	const unsigned vpf = m_tess->numVertices();
	const unsigned ipf = m_tess->numIndices();
	std::cout<<"ACC patch drawer\ncvs count: "<<vpf * numFace<<"("<<vpf<<" X "<<numFace<<")\n";
	std::cout<<"index count: "<<ipf * numFace<<"("<<ipf<<" X "<<numFace<<")\n";
}
//:~