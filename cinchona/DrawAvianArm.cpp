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
#include <gl_heads.h>

using namespace aphid; 

DrawAvianArm::DrawAvianArm()
{}

DrawAvianArm::~DrawAvianArm()
{}

void DrawAvianArm::drawSkeletonCoordinates()
{
	for(int i=0;i<6;++i) {
		drawCoordinateAt(&skeletonMatrix(i) );
	}
	
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
