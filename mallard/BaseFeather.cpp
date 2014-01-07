/*
 *  BaseFeather.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/21/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseFeather.h"

BaseFeather::BaseFeather() : m_quilly(0), m_uvDisplace(0), m_st(0), m_segementNormals(0)
{
	m_uv.set(4.f, 4.f);
}

BaseFeather::~BaseFeather() 
{
	if(m_quilly) delete[] m_quilly;
    if(m_uvDisplace) delete[] m_uvDisplace;
	if(m_st) delete[] m_st;
	if(m_segementNormals) delete[] m_segementNormals;
}

void BaseFeather::createNumSegment(short x)
{
	if(m_quilly) delete[] m_quilly;
    if(m_uvDisplace) delete[] m_uvDisplace;
	if(m_st) delete[] m_st;
	if(m_segementNormals) delete[] m_segementNormals;
    m_numSeg = x;
    m_quilly = new float[m_numSeg];
    m_uvDisplace = new Vector2F[(m_numSeg + 1) * 6];
	m_st = new Vector2F[(numSegment() + 1) * 7];
	m_segementNormals = new Vector3F[numSegment()];
}

short BaseFeather::numSegment() const
{
	return m_numSeg;
}

unsigned BaseFeather::numVaneVertices() const
{
	return (numSegment() + 1) * 6;
}
	
unsigned BaseFeather::numWorldP() const
{
	return (numSegment() + 1) * 7;
}

BoundingRectangle BaseFeather::getBoundingRectangle() const
{
	return m_brect;
}

float * BaseFeather::quilly()
{
    return m_quilly;
}

float * BaseFeather::getQuilly() const
{
     return m_quilly;
}

Vector2F * BaseFeather::uvDisplace()
{
	return m_uvDisplace;
}

Vector2F * BaseFeather::uvDisplaceAt(short seg, short side)
{
    return &m_uvDisplace[seg * 6 + 3 * side];
}

Vector2F * BaseFeather::getUvDisplaceAt(short seg, short side) const
{
    return &m_uvDisplace[seg * 6 + 3 * side];
}

Vector2F BaseFeather::baseUV() const
{
	return m_uv;
}

void BaseFeather::setBaseUV(const Vector2F & d)
{
	m_uv = d;
}

void BaseFeather::translateUV(const Vector2F & d)
{
	m_uv += d;
	for(unsigned i = 0; i < numWorldP(); i++)
		texcoord()[i] += d/32.f;

	m_brect.translate(d);
}

void BaseFeather::computeLength()
{
	m_shaftLength = 0.f;
	for(short i=0; i < m_numSeg; i++)
		m_shaftLength += m_quilly[i];
}

float BaseFeather::shaftLength() const
{
	return m_shaftLength;
}

float BaseFeather::width() const
{
	return m_brect.distance(0);
}

void BaseFeather::computeTexcoord()
{
	Vector2F puv = m_uv;
	float *q = quilly();
	int i, j;
	for(i=0; i <= numSegment(); i++) {
		*segmentQuillTexcoord(i) = puv;
		if(i < numSegment()) {
			puv += Vector2F(0.f, *q);
			q++;
		}
	}
	
	q = quilly();
	puv = m_uv;
	
	Vector2F pvane;
	for(i=0; i <= numSegment(); i++) {
		
		pvane = puv;
		Vector2F * vanes = uvDisplaceAt(i, 0);
		
		for(j = 0; j < 3; j++) {
			pvane += vanes[j];
			*segmentVaneTexcoord(i, 0, j) = pvane;
		}

		pvane = puv;
		vanes = getUvDisplaceAt(i, 1);
		
		for(j = 0; j < 3; j++) {
			pvane += vanes[j];
			*segmentVaneTexcoord(i, 1, j) = pvane;
		}
		
		if(i < numSegment()) {
			puv += Vector2F(0.f, *q);
			q++;
		}
	}
	
	for(i = 0; i < numWorldP(); i++) {

		texcoord()[i] /= 32.f;
	}
		
	computeBounding();
	computeLength();
}

void BaseFeather::computeBounding()
{
	m_brect.reset();
	for(unsigned i = 0; i < numWorldP(); i++) {
		Vector2F p = texcoord()[i] * 32.f;
		m_brect.update(p);
	}
}

void BaseFeather::simpleCreate(int ns)
{
    createNumSegment(ns);
	
    float * quill = quilly();
	int i;
	for(i = 0; i < ns; i++) {
		if(i < ns - 2)
			quill[i] = 3.f;
		else if(i < ns - 1)
			quill[i] = 1.7f;
		else
			quill[i] = .8f;
    }
	
	Vector2F * vanesR;
	Vector2F * vanesL;
	for(i = 0; i <= ns; i++) {
		vanesR = uvDisplaceAt(i, 0);
		vanesL = uvDisplaceAt(i, 1);
		
		if(i < ns - 2) {
			vanesR[0].set(.9f, .8f);
			vanesR[1].set(.8f, 1.1f);
			vanesR[2].set(.2f, 1.3f);
			
			vanesL[0].set(-.9f, .8f);
			vanesL[1].set(-.8f, 1.1f);
			vanesL[2].set(-.2f, 1.3f);
		}
		else if(i < ns - 1) {
			vanesR[0].set(.6f, .6f);
			vanesR[1].set(.4f, .5f);
			vanesR[2].set(.05f, .6f);
			
			vanesL[0].set(-.6f, .6f);
			vanesL[1].set(-.4f, .5f);
			vanesL[2].set(-.05f, .6f);
		}
		else if(i < ns) {
			vanesR[0].set(.3f, .3f);
			vanesR[1].set(.2f, .3f);
			vanesR[2].set(0.f, .4f);
			
			vanesL[0].set(-.3f, .3f);
			vanesL[1].set(-.2f, .3f);
			vanesL[2].set(0.f, .4f);
		}
		else {
			vanesR[0].set(0.f, .2f);
			vanesR[1].set(0.f, .1f);
			vanesR[2].set(0.f, .1f);
			
			vanesL[0].set(0.f, .2f);
			vanesL[1].set(0.f, .1f);
			vanesL[2].set(0.f, .1f);
		}
	}
	
	computeTexcoord();
}

void BaseFeather::changeNumSegment(int d)
{
	float * bakQuilly = new float[numSegment()];
    Vector2F *bakVaneVertices = new Vector2F[(numSegment() + 1) * 6];
	int i, j;
	for(i = 0; i < numSegment(); i++) {
		bakQuilly[i] = m_quilly[i];
	}
		
	for(i = 0; i < (numSegment() + 1) * 6; i++) {
		bakVaneVertices[i] = m_uvDisplace[i];
	}
		
	createNumSegment(numSegment() + d);
	
	const short numSeg = numSegment();
	
	if(d > 0) {
		for(i = 0; i < numSeg; i++) {
			if(i == 0) m_quilly[i] = bakQuilly[0];
			else m_quilly[i] = bakQuilly[i - 1];
		}
		for(i = 0; i <= numSeg; i++) {
			if(i == 0) {
				for(j = 0; j < 6; j++) {
					m_uvDisplace[i * 6 + j] = bakVaneVertices[j];
				}
			}
			else {
				for(j = 0; j < 6; j++) {
					m_uvDisplace[i * 6 + j] = bakVaneVertices[(i - 1) * 6 + j];
				}
			}
		}
		
	}
	else {
		for(i = 0; i < numSeg; i++) {
			if(i < numSeg -1) m_quilly[i] = bakQuilly[i];
			else m_quilly[i] = bakQuilly[i + 1];
		}
		for(i = 0; i <= numSeg; i++) {
			if(i < numSeg -1) {
				for(j = 0; j < 6; j++)
					m_uvDisplace[i * 6 + j] = bakVaneVertices[i * 6 + j] ;
			}
			else {
				for(j = 0; j < 6; j++)
					m_uvDisplace[i * 6 + j] = bakVaneVertices[(i + 1) * 6 + j] ;
			}
		}
	}
	
	delete[] bakQuilly;
	delete[] bakVaneVertices;
	
	computeTexcoord();
}

Vector2F * BaseFeather::texcoord()
{
	return m_st;
}

Vector2F * BaseFeather::segmentQuillTexcoord(short seg)
{
	return &m_st[seg * 7];
}

Vector2F * BaseFeather::segmentVaneTexcoord(short seg, short side, short idx)
{
	return &m_st[seg * 7 + 1 + side * 3 + idx];
}

float* BaseFeather::selectVertexInUV(const Vector2F & p, bool & yOnly, Vector2F & wp)
{
	float * r = 0;
	float minD = 10e8;
	yOnly = true;
	
	Vector2F puv;
	short seg, side, j;
	for(unsigned i = 1; i < numWorldP(); i++) {
		seg = i / 7;
		side = (i - seg*7 - 1) / 3;
		j = i - seg * 7 - 1 - side * 3;
		
		puv = m_st[i];
		puv *= 32.f;
		
		if(p.distantTo(puv) < minD) {
			minD = p.distantTo(puv);
			wp = puv;
			
			if(i % 7 == 0) {
				r = &quilly()[seg - 1];
				yOnly = true;
			}
			else {
				r = (float *)&uvDisplaceAt(seg, side)[j];
				yOnly = false;
			}
		}
	}
	
	return r;
}

Vector3F * BaseFeather::normal(unsigned seg)
{
	return &m_segementNormals[seg];
}
