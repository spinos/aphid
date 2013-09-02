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
	m_tess = new Tessellator;
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

void BezierDrawer::drawAccPatchMesh(AccPatchMesh * mesh)
{setWired(1);
	const unsigned numFace = mesh->numPatches();
	AccPatch* bez = mesh->beziers();
	for(unsigned i = 0; i < numFace; i++) {
		//drawBezierCage(&bez[i]);
		BoundingBox b = bez[i].controlBBox();
		box(b);
	}
}
//:~