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

BezierDrawer::BezierDrawer()
{
	m_tagName = "unknown";
	m_tess = new Tessellator;
}

BezierDrawer::~BezierDrawer()
{
	delete m_tess;
}

void BezierDrawer::rebuildBuffer(AccPatchMesh * mesh)
{
	std::clog<<"buffering bezier patch...";
	AccPatch* bez = mesh->beziers();
	
	m_tess->setNumSeg(4);
	const unsigned numFace = mesh->getNumFaces();
	const unsigned vpf = m_tess->numVertices();
	const unsigned ipf = m_tess->numIndices();
	
	createBuffer(vpf * numFace, ipf * numFace);
	std::clog<<" n points "<<numPoints();
	
	Vector3F * cv = vertices();
	Vector3F * normal = normals();
	float * uv = texcoords();
	Float3 * col = colors();
	
	unsigned curP = 0, curI = 0, faceStart;
	unsigned i, j;
	for(i = 0; i < numFace; i++) {
		m_tess->evaluate(bez[i]);
		Vector3F *pop = m_tess->_positions;
		Vector3F *nor = m_tess->_normals;
		Vector3F * texcoord = m_tess->_texcoords;
		int *idr = m_tess->getVertices();
		for(j = 0; j < vpf; j++) {
			cv[curP] = pop[j];
			normal[curP] = nor[j];
			col[curP] = Float3(.75f, .75f, .75f);
			uv[curP * 2] = texcoord[j].x;
			uv[curP * 2 + 1] = texcoord[j].y;
			
			curP++;
		}
		faceStart = vpf * i;
		for(j = 0; j < ipf; j++) {
			indices()[curI] = faceStart + idr[j];
			curI++;
		}
	}
	
	if(m_tagName != "unknown") tagColor(mesh);
	std::clog<<"...done!\n";
}

void BezierDrawer::drawBezierPatch(BezierPatch * patch)
{
	int seg = 4;
	m_tess->setNumSeg(seg);
	m_tess->evaluate(*patch);
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

void BezierDrawer::useTag(const std::string & name)
{
	m_tagName = name;
}

void BezierDrawer::tagColor(AccPatchMesh * mesh)
{
	char * g = mesh->perFaceTag(m_tagName);
	const unsigned vpf = m_tess->numVertices();
	const unsigned numFace = mesh->getNumFaces();
	unsigned curP = 0;
	unsigned i, j;
	for(i = 0; i < numFace; i++) {
		for(j = 0; j < vpf; j++) {
			if(g[i] == 1) colors()[curP] = Float3(.75f, .75f, .75f);
			else colors()[curP] = Float3(.4f, .4f, .4f);
			curP++;
		}
	}
}
//:~