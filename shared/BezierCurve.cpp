/*
 *  BezierCurve.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 5/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BezierCurve.h"

BezierCurve::BezierCurve() {}
BezierCurve::~BezierCurve() {}

Vector3F BezierCurve::interpolate(float param) const
{	
	unsigned seg = segmentByParameter(param);
	Vector3F p[4];
	
	calculateCage(seg, p);
	float t = param * numSegments() - seg;
	
	return calculateBezierPoint(t, p);
}

void BezierCurve::calculateCage(unsigned seg, Vector3F *p) const
{
	if(seg == 0) p[0] = m_cvs[0];
	else p[0] = (m_cvs[seg] * 2.f + m_cvs[seg - 1] + m_cvs[seg + 1]) * .25f;

	if(seg >= numSegments() - 1) p[3] = m_cvs[numSegments()];
	else p[3] = (m_cvs[seg + 1] * 2.f + m_cvs[seg] + m_cvs[seg + 2]) * .25f;

	p[1] = (m_cvs[seg + 1] * 0.5f + m_cvs[seg] * 0.5f) * .5f + p[0] * .5f;
	p[2] = (m_cvs[seg] * 0.5f + m_cvs[seg + 1] * 0.5f) * .5f + p[3] * .5f;
}

Vector3F BezierCurve::calculateBezierPoint(float t, Vector3F * data) const
{
	float u = 1.f - t;
	float tt = t * t;
	float uu = u*u;
	float uuu = uu * u;
	float ttt = tt * t;

	Vector3F p0 = data[0];
	Vector3F p1 = data[1];
	Vector3F p2 = data[2];
	Vector3F p3 = data[3];

	Vector3F p = p0 * uuu; //first term
	p += p1 * 3.f * uu * t; //second term
	p += p2 * 3.f * u * tt; //third term
	p += p3 * ttt; //fourth term
	return p;
}

void BezierCurve::getAccSegmentCurves(BezierCurve * dst) const
{
    unsigned i, j;
    unsigned v[4];
    for(j = 0; j < numSegments(); j++) {
        v[0] = j-1;
        v[1] = j;
        v[2] = j+1;
        v[3] = j+2;
        if(j==0) v[0] = 0;
        if(j==numSegments()-1) v[3] = j+1;
        
        if(j==0) dst[j].m_cvs[0] = m_cvs[v[0]];
        else dst[j].m_cvs[0] = m_cvs[v[0]] * .25f + m_cvs[v[1]] * .5f + m_cvs[v[2]] * .25f ;
        dst[j].m_cvs[1] = (dst[j].m_cvs[0] * 5.f + m_cvs[v[1]] * 2.f + m_cvs[v[2]] * 2.f) * (1.f / 9.f);
        
        if(j==numSegments()-1) dst[j].m_cvs[3] = m_cvs[v[3]];
        else dst[j].m_cvs[3] = m_cvs[v[1]] * .25f + m_cvs[v[2]] * .5f + m_cvs[v[3]] * .25f ;
        dst[j].m_cvs[2] = (dst[j].m_cvs[3] * 5.f + m_cvs[v[1]] * 2.f + m_cvs[v[2]] * 2.f) * (1.f / 9.f);
    }
}

void BezierCurve::getAccSegmentSpline(unsigned i, SimpleBezierSpline & sp) const
{
    unsigned v[4];
    v[0] = i-1;
    v[1] = i;
    v[2] = i+1;
    v[3] = i+2;
    if(i<1) v[0] = 0;
    if(i==numSegments()-1) v[3] = i+1;
    
    if(i==0) sp.cv[0] = m_cvs[v[0]];
    else sp.cv[0] = m_cvs[v[0]] * .25f + m_cvs[v[1]] * .5f + m_cvs[v[2]] * .25f ;
    sp.cv[1] = (sp.cv[0] * 5.f + m_cvs[v[1]] * 2.f + m_cvs[v[2]] * 2.f) * (1.f / 9.f);
    
    if(i==numSegments()-1) sp.cv[3] = m_cvs[v[3]];
    else sp.cv[3] = m_cvs[v[1]] * .25f + m_cvs[v[2]] * .5f + m_cvs[v[3]] * .25f ;
    sp.cv[2] = (sp.cv[3] * 5.f + m_cvs[v[1]] * 2.f + m_cvs[v[2]] * 2.f) * (1.f / 9.f);
}

float BezierCurve::distanceToPoint(const Vector3F & toP, Vector3F & closestP) const
{
    float minD = 1e8;
    const unsigned ns = numSegments();
    for(unsigned i=0; i < ns; i++) {
        SimpleBezierSpline sp;
        getAccSegmentSpline(i, sp);   
        distanceToPoint(sp, toP, minD, closestP);
    }
    return minD;
}

void BezierCurve::distanceToPoint(SimpleBezierSpline & spline, const Vector3F & pnt, float & minDistance, Vector3F & closestP) const
{
    float paramMin = 0.f;
    float paramMax = 1.f;
    Vector3F line[2];
    
    line[0] = spline.calculateBezierPoint(paramMin);
    line[1] = spline.calculateBezierPoint(paramMax);
    
    Vector3F pol;
    float t;
    for(;;) {
        float d = closestDistanceToLine(line, pnt, pol, t);
        
        const float tt = paramMin * (1.f - t) + paramMax * t;
        
// end of line
        if((tt <= 0.f || tt >= 1.f) && paramMax - paramMin < 0.1f) {
            if(d < minDistance) {
                minDistance = d;
                closestP = pol;
            }
            break;
        }
        
        const float h = .5f * (paramMax - paramMin);

// small enought        
        if(h < 1e-3) {
            if(d < minDistance) {
                minDistance = d;
                closestP = pol;
            }
            break;
        }
        
        if(t > .5f)
            paramMin = tt - h * .5f;
            
        else
            paramMax = tt + h * .5f;
            
        line[0] = spline.calculateBezierPoint(paramMin);
        line[1] = spline.calculateBezierPoint(paramMax);
    }
}

