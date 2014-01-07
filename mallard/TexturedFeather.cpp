/*
 *  TexturedFeather.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "TexturedFeather.h"
#include "BaseVane.h"
#include <AdaptableStripeBuffer.h>

ZEXRImage TexturedFeather::ColorTextureFile;

TexturedFeather::TexturedFeather() 
{
	m_vane = new BaseVane[2];
	m_stripe = new AdaptableStripeBuffer;
	m_resShaft = 10;
	m_resBarb = 9;
}

TexturedFeather::~TexturedFeather() 
{
	delete[] m_vane;
	delete m_stripe;
}

void TexturedFeather::computeTexcoord()
{
	BaseFeather::computeTexcoord();
	
	if(m_vane[0].gridU() != numSegment())
		m_vane[0].create(numSegment(), 3);
	if(m_vane[1].gridU() != numSegment())
		m_vane[1].create(numSegment(), 3);
	
	const short numSeg = numSegment();
	for(short i = 0; i <= numSeg; i++) {	
		*m_vane[0].railCV(i, 0) = Vector3F(*segmentQuillTexcoord(i));
		*m_vane[1].railCV(i, 0) = Vector3F(*segmentQuillTexcoord(i));
		for(short j = 0; j < 3; j++) {
			*m_vane[0].railCV(i, j + 1) = Vector3F(*segmentVaneTexcoord(i, 0, j));
			*m_vane[1].railCV(i, j + 1) = Vector3F(*segmentVaneTexcoord(i, 1, j));
		}
	}

	m_vane[0].computeKnots();
	m_vane[1].computeKnots();
}

void TexturedFeather::translateUV(const Vector2F & d)
{
	BaseFeather::translateUV(d);
	const short numSeg = numSegment();
	for(short i = 0; i <= numSeg; i++) {	
		*m_vane[0].railCV(i, 0) = Vector3F(*segmentQuillTexcoord(i));
		*m_vane[1].railCV(i, 0) = Vector3F(*segmentQuillTexcoord(i));
		for(short j = 0; j < 3; j++) {
			*m_vane[0].railCV(i, j + 1) = Vector3F(*segmentVaneTexcoord(i, 0, j));
			*m_vane[1].railCV(i, j + 1) = Vector3F(*segmentVaneTexcoord(i, 1, j));
		}
	}
}

BaseVane * TexturedFeather::uvVane(short side) const
{
	return &m_vane[side];
}

Vector3F * TexturedFeather::uvVaneCoord(short u, short v, short side)
{
	return m_vane[side].railCV(u, v);
}

void TexturedFeather::sampleColor(unsigned gridU, unsigned gridV, Vector3F * dst)
{
	if(!ColorTextureFile.isOpened()) {
		const unsigned ns = (gridU + 1) * (gridV + 1) * 2;
		for(unsigned i = 0; i < ns; i++) dst[i].set(1.f, 1.f, 1.f);
		return;
	}
	
	const float du = 1.f/(float)gridU;
	const float dv = 1.f/(float)gridV;
	
	unsigned acc = 0;
	Vector3F coord;
	for(unsigned i = 0; i <= gridU; i++) {
		m_vane[0].setU(du*i);
		for(unsigned j = 0; j <= gridV; j++) {
			m_vane[0].pointOnVane(dv * j, coord);
			ColorTextureFile.sample(coord.x, coord.y, 3, (float *)&dst[acc]);
			acc++;
		}
	}
	
	for(unsigned i = 0; i <= gridU; i++) {
		m_vane[1].setU(du*i);
		for(unsigned j = 0; j <= gridV; j++) {
			m_vane[1].pointOnVane(dv * j, coord);
			ColorTextureFile.sample(coord.x, coord.y, 3, (float *)&dst[acc]);
			acc++;
		}
	}
}

void TexturedFeather::sampleColor(float lod)
{
	m_stripe->create(m_resShaft * m_vane[0].gridU() * 2, m_resBarb + 1);

	const unsigned nu = m_vane[0].gridU() * (2 + (resShaft() - 2) * lod);
	const unsigned nv = 3 + (m_resBarb - 3) * lod;
	m_stripe->begin();
	
	sampleColor(nu, nv, 0);
	sampleColor(nu, nv, 1);
}

void TexturedFeather::sampleColor(unsigned nu, unsigned nv, int side)
{
	const float du = 1.f/(float)nu;
	const float dv = 1.f/(float)nv;

	for(unsigned i = 0; i < nu; i++) {
		*m_stripe->currentNumCvs() = nv + 1;
		
		Vector3F * coord = m_stripe->currentPos();
		Vector3F * col = m_stripe->currentCol();
		m_vane[side].setU(du*i);
		for(unsigned j = 0; j <= nv; j++) {
			if(!ColorTextureFile.isOpened()) {
				col[j].set(1.f, 1.f, 1.f);
			}
			else {
				m_vane[side].pointOnVane(dv * j, coord[j]);
				ColorTextureFile.sample(coord[j].x, coord[j].y, 3, (float *)&col[j]);
			}
		}
		
		m_stripe->next();
	}
}

void TexturedFeather::setResShaft(unsigned resShaft)
{
	m_resShaft = resShaft;
	if(m_resShaft < 2) m_resShaft = 2;
}

void TexturedFeather::setResBarb(unsigned resBarb)
{
	m_resBarb = resBarb;
	if(m_resBarb < 3) m_resShaft = 3;
}

unsigned TexturedFeather::resShaft() const
{
	return m_resShaft;
}

unsigned TexturedFeather::resBarb() const
{
	return m_resBarb;
}

AdaptableStripeBuffer * TexturedFeather::stripe()
{
	return m_stripe;
}

unsigned TexturedFeather::numStripe() const
{
	return m_stripe->numStripe();
}

unsigned TexturedFeather::numStripePoints() const
{
	return m_stripe->numPoints();
}
