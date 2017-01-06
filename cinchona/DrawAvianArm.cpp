/*
 *  DrawAvianArm.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawAvianArm.h"
#include "Ligament.h"
#include <math/Vector3F.h>
#include <math/Matrix44F.h>
#include "FeatherMesh.h"
#include "FeatherObject.h"
#include "FeatherDeformParam.h"
#include "FeatherDeformer.h"
#include "WingRib.h"
#include "WingSpar.h"
#include <gl_heads.h>

using namespace aphid; 

DrawAvianArm::DrawAvianArm()
{}

DrawAvianArm::~DrawAvianArm()
{}

void DrawAvianArm::drawSkeletonCoordinates()
{
	drawCoordinateAt(principleMatrixR() );
	
}

void DrawAvianArm::drawLigaments()
{
	float m[16];
	principleMatrixR()->glMatrix(m);
	glPushMatrix();
	glMultMatrixf(m);
	drawLigament(leadingLigament() );
	drawLigament(trailingLigament() );
	glPopMatrix();
	
}

void DrawAvianArm::drawFeathers()
{
    const int n = numFeathers();
    if(n<1) {
        return;   
    }
    
    glColor3f(.45f, .55f, .65f);
    
    float m[16];
	principleMatrixR()->glMatrix(m);
	glPushMatrix();
	glMultMatrixf(m);
	
	for(int i=0;i<n;++i) {
	    const FeatherObject * f = feather(i);
	    
	    f->glMatrix(m);
	    glPushMatrix();
	    glMultMatrixf(m);
	    
	    drawFeatherMesh(f->mesh(), f->deformer() );
	    
	    glPopMatrix();
	}
	glPopMatrix();
}

void DrawAvianArm::drawLigament(const Ligament & lig)
{
	const int & np = lig.numPieces();
	glBegin(GL_LINE_STRIP);
	for(int j=0;j<np;++j) {
		for(int i=0;i<50;++i) {
			const Vector3F p = lig.getPoint(j, 0.02*i);
			glVertex3fv((const float *)&p);
		}
	}
	glEnd();
	
}

void DrawAvianArm::drawFeatherLeadingEdge(const FeatherMesh * mesh,
										const FeatherDeformer * deformer)
{
	glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)deformer->deformedLeadingEdgePoints() );
    glDrawArrays(GL_LINE_STRIP, 0, mesh->numLeadingEdgeVertices() );
    
    glDisableClientState(GL_VERTEX_ARRAY);
}

void DrawAvianArm::drawFeatherMesh(const FeatherMesh * mesh,
										const FeatherDeformer * deformer)
{
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, 0, (const GLfloat*)deformer->deformedNormals());
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)deformer->deformedPoints() );
    glDrawElements(GL_TRIANGLES, mesh->numIndices(), GL_UNSIGNED_INT, mesh->indices());
    
    glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void DrawAvianArm::drawFeatherOrietations()
{
	FeatherDeformParam * param = featherDeformParameter();
	const Vector3F localOffset(0,5,0);
	float m[16];
	principleMatrixR()->glMatrix(m);
	glPushMatrix();
	glMultMatrixf(m);
	
	Matrix44F locm = *secondDigitMatirxR();
	locm.setTranslation(secondDigitEndPosition() );
	locm *= *invPrincipleMatrixR();
	locm.setTranslation(trailingLigament().getPoint(2, 0.99f) + localOffset);
	
	drawFlatArrowTandem(&locm, &param->rotation(3) );
	
	locm = *midsection1MarixR();
	locm.setTranslation(trailingLigament().getPoint(1, 0.99f) + localOffset);
	
	drawFlatArrowTandem(&locm, &param->rotation(2) );
	
	locm = *midsection0MarixR();
	locm.setTranslation(trailingLigament().getPoint(0, 0.99f) + localOffset);
	
	drawFlatArrowTandem(&locm, &param->rotation(1) );
	
	locm = *inboardMarixR();
	locm.setTranslation(shoulderPosition() + localOffset);
	
	locm *= *invPrincipleMatrixR();
	drawFlatArrowTandem(&locm, &param->rotation(0) );
	
	glPopMatrix();
	
}

void DrawAvianArm::drawFeatherLeadingEdges()
{
	const int n = numFeathers();
    if(n<1) {
        return;   
    }
    
    glColor3f(.35f, .15f, .85f);
    
    float m[16];
	principleMatrixR()->glMatrix(m);
	glPushMatrix();
	glMultMatrixf(m);
	
	for(int i=0;i<n;++i) {
	    const FeatherObject * f = feather(i);
	    
	    f->glMatrix(m);
	    glPushMatrix();
	    glMultMatrixf(m);
	    
	    drawFeatherLeadingEdge(f->mesh(), f->deformer() );
	    
	    glPopMatrix();
	}
	glPopMatrix();

}

static const float sRibLineParam[21] = {
0.f, 0.1f, 0.2f, 0.3f, 0.4f, .5f, .6f, .7f, .8f, .9f, .999f, 
-.1f, -.2f, -.3f, -.4f, -.5f, -.6f, -.7f, -.8f, -.9f, -1.f
};

void DrawAvianArm::drawRibs()
{
	float m[16];
	principleMatrixR()->glMatrix(m);
	glPushMatrix();
	glMultMatrixf(m);
	
	Vector3F p;
	
	for(int i=0;i<5;++i) {
		const WingRib * r = rib(i);
		
		glBegin(GL_LINE_STRIP);
		
		for(int j =0;j<21;++j) {
			r->getPoint(p, sRibLineParam[j]);
			glVertex3fv((const float *)&p);
		
		}
		glEnd();
	}
	
	glPopMatrix();
}

void DrawAvianArm::drawSpars()
{
	float m[16];
	principleMatrixR()->glMatrix(m);
	glPushMatrix();
	glMultMatrixf(m);
	
	Vector3F p;
	
	for(int i=0;i<4;++i) {
		const WingSpar * s = spar(i);
		
		glBegin(GL_LINE_STRIP);
		
		for(int j =0;j<100;++j) {
			
			s->getPoint(p, j);
			
			glVertex3fv((const float *)&p);
		
		}
		glEnd();
	}
	
	glPopMatrix();

}
