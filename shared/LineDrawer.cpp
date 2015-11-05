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
LineDrawer::LineDrawer() {}
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

void LineDrawer::frustum(const Frustum * f)
{	
	glBegin(GL_LINES);
	Vector3F p = f->X(0);
	glVertex3fv((float *)&p);
	p = f->X(1);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(2);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(3);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(0);
	glVertex3fv((float *)&p);
	
	p = f->X(4);
	glVertex3fv((float *)&p);
	p = f->X(5);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(6);
	glVertex3fv((float *)&p);
	glVertex3fv((float *)&p);
	p = f->X(7);
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
//:~