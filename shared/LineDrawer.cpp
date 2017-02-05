/*
 *  LineDrawer.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "LineDrawer.h"
#include <LineBuffer.h>
#include <AdaptableStripeBuffer.h>
#include <BaseCurve.h>
#include <BezierCurve.h>
#include <geom/ConvexShape.h>

namespace aphid {

float LineDrawer::DigitLineP[10][8][2] = {
{{0.3f, 0.1f}, {0.6f, 0.1f}, {0.63f, 0.13f}, {0.73f, 0.9f}, {0.43f, 0.9f}, {0.4f, 0.87f}, {0.3f, 0.13f}, {0.3f, 0.1f} },
{{0.3f, 0.1f}, {0.31f, 0.1f}, {0.33f, 0.13f}, {0.43f, 0.9f}, {0.43f, 0.9f}, {0.4f, 0.87f}, {0.3f, 0.13f}, {0.3f, 0.1f} },
{{0.3f, 0.86f}, {0.33f, 0.9f}, {0.5f, 0.9f}, {0.6f, 0.7f}, {0.55f, 0.66f}, {0.21f, 0.23f}, {0.2f, 0.1f}, {0.6f, 0.1f} },
{{0.22f, 0.9f}, {0.6f, 0.9f}, {0.62f, 0.88f}, {0.57f, 0.65f}, {0.3f, 0.55f}, {0.55f, 0.5f}, {0.5f, 0.2f}, {0.2f, 0.2f} },
{{0.23f, 0.9f}, {0.2f, 0.6f}, {0.55f, 0.56f}, {0.6f, 0.9f}, {0.55f, 0.56f}, {0.6f, 0.56f}, {0.55f, 0.56f}, {0.5f, 0.1f} },
{{0.65f, 0.9f}, {0.3f, 0.9f}, {0.27f, 0.55f}, {0.55f, 0.55f}, {0.6f, 0.5f}, {0.58f, 0.13f}, {0.55f, 0.1f}, {0.2f, 0.1f} },
{{0.3f, 0.9f}, {0.27f, 0.55f}, {0.55f, 0.55f}, {0.6f, 0.5f}, {0.58f, 0.13f}, {0.55f, 0.1f}, {0.2f, 0.1f}, {0.25f, 0.52f} },
{{0.2f, 0.9f}, {0.6f, 0.9f}, {0.62f, 0.89f}, {0.3f, 0.1f}, {0.29f, 0.1f}, {0.61f, 0.89f}, {0.59f, 0.88f}, {0.2f, 0.88f} },
{{0.2f, 0.1f}, {0.23f, 0.9f}, {0.63f, 0.9f}, {0.62f, 0.56f}, {0.22f, 0.56f}, {0.62f, 0.56f}, {0.6f, 0.1f}, {0.2f, 0.1f} },
{{0.22f, 0.56f}, {0.23f, 0.9f}, {0.63f, 0.9f}, {0.62f, 0.56f}, {0.22f, 0.56f}, {0.62f, 0.56f}, {0.6f, 0.1f}, {0.2f, 0.1f} }	
};

int LineDrawer::DigitM[9] = {
10000000,
1000000,
100000,
10000,
1000,
100,
10,
1,
0
};

LineDrawer::LineDrawer() 
{ m_alignDir = Vector3F::ZAxis; }

LineDrawer::~LineDrawer() {}

void LineDrawer::drawLineBuffer(LineBuffer * line) const
{
    if(line->numVertices() < 2) return;
    glBegin(GL_LINES);
    for(unsigned i = 0; i < line->numVertices(); i++)
    {
        Vector3F p = line->vertices()[i];
        glVertex3f(p.x , p.y, p.z);
    }
    glEnd();
}

void LineDrawer::lines(const std::vector<Vector3F> & vs)
{
	glBegin(GL_LINES);
	std::vector<Vector3F>::const_iterator it = vs.begin();
	for(; it != vs.end(); ++it)
		glVertex3fv((float *)&(*it));
	glEnd();
}

void LineDrawer::lineStripes(const unsigned & num, unsigned * nv, Vector3F * vs) const
{
	unsigned acc = 0;
	for(unsigned i = 0; i < num; i++) {
		glBegin(GL_LINE_STRIP);
		for(unsigned j = 0; j < nv[i]; j++) {
			glVertex3fv((float *)&vs[acc]);
			acc++;
		}
		glEnd();
	}
}

void LineDrawer::lineStripes(const unsigned & num, unsigned * nv, Vector3F * vs, Vector3F * cs) const
{
	unsigned acc = 0;
	for(unsigned i = 0; i < num; i++) {
		glBegin(GL_LINE_STRIP);
		for(unsigned j = 0; j < nv[i]; j++) {
			glColor3fv((float *)&cs[acc]);
			glVertex3fv((float *)&vs[acc]);
			acc++;
		}
		glEnd();
	}
}

void LineDrawer::stripes(AdaptableStripeBuffer * data, const Vector3F & eyeDirection) const
{
	const unsigned ns = data->numStripe();
	unsigned * ncv = data->numCvs();
	Vector3F * pos = data->pos();
	Vector3F * col = data->col();
	float * w = data->width();
	Vector3F seg, nseg, straggle, q, q0, q1;
	glBegin(GL_QUADS);
	for(unsigned i = 0; i < ns; i++) {
		for(unsigned j = 0; j < ncv[i] - 1; j++) {
			seg = pos[j+1] - pos[j];
			nseg = seg.normal();
			straggle = nseg.cross(eyeDirection);
			straggle.normalize();
			
			glColor3fv((float *)&col[j]);
			
			if(j< 1) {
				q = pos[j] - straggle * w[j];
				glVertex3fv((float *)&q);
			
				q = pos[j] + straggle * w[j];
				glVertex3fv((float *)&q);
			}
			else {
				glVertex3fv((float *)&q0);
				glVertex3fv((float *)&q1);
			}
			
			glColor3fv((float *)&col[j+1]);
			
			q = pos[j+1] + straggle * w[j+1];
			glVertex3fv((float *)&q);
			
			q1 = q;
			
			q = pos[j+1] - straggle * w[j+1];
			glVertex3fv((float *)&q);
			
			q0 = q;
		}
		pos += ncv[i];
		col += ncv[i];
		w += ncv[i];
	}
	glEnd();
}

void LineDrawer::linearCurve(const BaseCurve & curve) const
{
    glDisable(GL_DEPTH_TEST);
	float t;
	Vector3F p;
	glBegin(GL_LINE_STRIP);
	for(unsigned i = 0; i < curve.numVertices(); i++) {
		p = curve.getCv(i);
		//t = curve.getKnot(i);
		//setColor(1.f - t, 0.f, t);
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
	glEnable(GL_DEPTH_TEST);
}

void LineDrawer::smoothCurve(const BezierCurve & curve, short deg) const
{
	const unsigned ns = curve.numSegments();
	unsigned i;
	for(i = 0; i < ns; i++) {
	    BezierSpline sp;
	    curve.getSegmentSpline(i, sp);
		smoothCurve(sp, deg);
	}
}

void LineDrawer::smoothCurve(const BezierSpline & sp, short deg) const
{
	const float delta = 1.f / (float)deg;
	Vector3F p0, p;
	glBegin(GL_LINES);
	short j;
	p0 = sp.calculateBezierPoint(0.f);
	for(j=1; j <= deg; j++) {
		glVertex3fv((GLfloat *)&p0);
		p = sp.calculateBezierPoint(delta * j);
		glVertex3fv((GLfloat *)&p);
		p0 = p;
	}
	glEnd();
}

void LineDrawer::drawNumber(int x, const Vector3F & p, float scale) const
{
	glPushMatrix();
	
	Matrix44F mat;
    mat.setTranslation(p);
    mat.setFrontOrientation(alignDir() );
	mat.scaleBy(scale);
	useSpace(mat);
	
	if(x == 0) {
		drawDigit(0);
		glPopMatrix();
		return;
	}
	int m = x;
/// limit of n digits
	int n = 0;
	while(n<8) {
		int r = m / DigitM[n];
		if(x>= DigitM[n]) {
			drawDigit(r);
			m -= r * DigitM[n];
			
			glTranslatef(.6f, 0.f, 0.f);
		}
		n++;
	}
	glPopMatrix();
}

void LineDrawer::drawDigit(int d) const
{
	glBegin(GL_LINE_STRIP);
	for(int i=0; i< 8; ++i) {
		glVertex3f(DigitLineP[d][i][0], DigitLineP[d][i][1], 0.f);
	}
	glEnd();
}

void LineDrawer::setAlignDir(const Vector3F & v)
{ m_alignDir = v; }

const Vector3F & LineDrawer::alignDir() const
{ return m_alignDir; }

void LineDrawer::frustum(const cvx::Frustum * f)
{	
	glBegin(GL_LINES);
	Vector3F p = f->X(0);
	glVertex3fv((float *)&p);
	p = f->X(1);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(3);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(2);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(0);
	glVertex3fv((float *)&p);
	
	p = f->X(4);
	glVertex3fv((float *)&p);
	p = f->X(5);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(7);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(6);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(4);
	glVertex3fv((float *)&p);
	
	glVertex3fv((float *)&p);
	p = f->X(0);
	glVertex3fv((float *)&p);
	
	p = f->X(5);
	glVertex3fv((float *)&p);
	p = f->X(1);
	glVertex3fv((float *)&p);
	
	p = f->X(6);
	glVertex3fv((float *)&p);
	p = f->X(2);
	glVertex3fv((float *)&p);
	
	p = f->X(7);
	glVertex3fv((float *)&p);
	p = f->X(3);
	glVertex3fv((float *)&p);
	glEnd();
}

}
//:~