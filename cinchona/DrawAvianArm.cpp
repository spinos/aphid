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
#include "FeatherGeomParam.h"
#include "Geom1LineParam.h"
#include <gl_heads.h>

using namespace aphid; 

DrawAvianArm::DrawAvianArm()
{}

DrawAvianArm::~DrawAvianArm()
{}

void DrawAvianArm::drawSkeletonCoordinates()
{
	drawCoordinateAt(principleMatrixR() );
	drawCoordinateAt(handMatrixR() );
	
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
	
	drawCoordinateTandem(&locm, &param->rotation(3) );
	drawFlatArrowTandem(&locm, &param->rotation(3) );
	
	locm = *midsection1MarixR();
/// local to hand
//	Matrix33F m1rot = locm.rotation() * handMatrixR()->rotation();
//	locm.setRotation(m1rot); 
	locm.setTranslation(trailingLigament().getPoint(1, 0.99f) + localOffset);
	
	drawCoordinateTandem(&locm, &param->rotation(2) );
	drawFlatArrowTandem(&locm, &param->rotation(2) );
	
	locm = *midsection0MarixR();
	locm.setTranslation(trailingLigament().getPoint(0, 0.99f) + localOffset);
	
	drawCoordinateTandem(&locm, &param->rotation(1) );
	drawFlatArrowTandem(&locm, &param->rotation(1) );
	
	locm = *inboardMarixR();
	locm.setTranslation(shoulderPosition() + localOffset);
	
	locm *= *invPrincipleMatrixR();
	
	drawCoordinateTandem(&locm, &param->rotation(0) );
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

void DrawAvianArm::drawFeatherContours()
{
	float m[16];
	principleMatrixR()->glMatrix(m);
	glPushMatrix();
	glMultMatrixf(m);
	
	int it = 0;
	for(int i=0;i<5;++i) {
		Geom1LineParam * line = featherGeomParameter()->line(i);
		drawFeatherLineContour(line, it);
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

void DrawAvianArm::drawFeatherLineContour(const Geom1LineParam * line,
									int & it)
{
	int n = line->numGeomsM1();
	Vector3F samp[2];
	glBegin(GL_LINE_STRIP);
	for(int i=0;i<n;++i) {
	
		const FeatherObject * f = feather(it);
		f->getEndPoints(samp);
		
		glVertex3fv((const float *)&samp[0]);
		it++;
	}
	glEnd();
}

void DrawAvianArm::drawPlane()
{
	glDisable(GL_LIGHTING);
	int it = 0;
	Geom1LineParam * line = featherGeomParameter()->line(0);
	Vector3F samp[2];
	Vector3F dev0[2];
	Vector3F dev1[2];
	
	glBegin(GL_LINES);
	
	int n = 0;
	const int nseg = line->numSegments();
	for(int i=0;i<nseg;++i) {
		const int nf = line->numFeatherOnSegment(i);
		n += nf - 1;
	}
	    
		
	    for(int i=0;i<n;++i) {
			const FeatherObject * f = feather(it);

	        f->getEndPoints(samp);
			
			if(i==0) {
				feather(it+1)->getEndPoints(dev1);
				dev1[0] -= samp[0];
				dev1[1] -= samp[1];
				dev0[0] = dev1[0];
				dev0[1] = dev1[1];
			} else if(i+1 == n) {
				feather(it-1)->getEndPoints(dev0);
				dev0[0] = samp[0] - dev0[0];
				dev0[1] = samp[1] - dev0[1];
				dev1[0] = dev0[0];
				dev1[1] = dev0[1];
			} else {
				feather(it-1)->getEndPoints(dev0);
				feather(it+1)->getEndPoints(dev1);
				dev0[0] = samp[0] - dev0[0];
				dev0[1] = samp[1] - dev0[1];
				dev1[0] -= samp[0];
				dev1[1] -= samp[1];
			}
		
			glColor3f(1,0,0);
			glVertex3fv((const float *)&samp[0]);
			glVertex3fv((const float *)&(samp[0]+dev0[0]) );
			
			glVertex3fv((const float *)&samp[1]);
			glVertex3fv((const float *)&(samp[1]+dev0[1]) );
			
			glColor3f(1,1,0);
			glVertex3fv((const float *)&samp[0]);
			glVertex3fv((const float *)&(samp[0]+dev1[0]) );
			
			glVertex3fv((const float *)&samp[1]);
			glVertex3fv((const float *)&(samp[1]+dev1[1]) );

			
			
			Matrix44F invRot = *f;
			invRot.inverse();
			dev0[0] = invRot.transformAsNormal(dev0[0]);
			
			if(dev0[0].y > 0.f) {
				Vector3F zdir = f->getFront();
			glColor3f(0,0,1);
			glVertex3fv((const float *)&samp[0]);
			glVertex3fv((const float *)&(samp[0]+zdir) );
			
			Vector3F ydir = f->getUp();
			glColor3f(0,1,0);
			glVertex3fv((const float *)&samp[0]);
			glVertex3fv((const float *)&(samp[0]+ydir) );
			}
			
			dev0[1] = invRot.transformAsNormal(dev0[1]);
			
			if(dev0[1].y > 0.f) {
				Vector3F zdir = f->getFront();
			glColor3f(0,0,1);
			glVertex3fv((const float *)&samp[1]);
			glVertex3fv((const float *)&(samp[1]+zdir) );
			
			Vector3F ydir = f->getUp();
			glColor3f(0,1,0);
			glVertex3fv((const float *)&samp[1]);
			glVertex3fv((const float *)&(samp[1]+ydir) );
			}
			
	        it++;
	    }
	
	
	glEnd();
}
