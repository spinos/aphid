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
	createVanes();
}

void TexturedFeather::translateUV(const Vector2F & d)
{
	BaseFeather::translateUV(d);
	shapeVanes();
}

void TexturedFeather::createVanes()
{
	if(type() == 0) {
		m_vane[0].create(numSegment(), 3);
		m_vane[1].create(numSegment(), 3);
	}
	else {
		m_vane[0].create(3, numSegment());
		m_vane[1].create(3, numSegment());
	}
	shapeVanes();
	m_vane[0].computeKnots();
	m_vane[1].computeKnots();
}

void TexturedFeather::shapeVanes()
{
	const short numSeg = numSegment();
	if(type() == 0) {
		for(short i = 0; i <= numSeg; i++) {	
			*m_vane[0].railCV(i, 0) = Vector3F(*segmentQuillTexcoord(i));
			*m_vane[1].railCV(i, 0) = Vector3F(*segmentQuillTexcoord(i));
			for(short j = 0; j < 3; j++) {
				*m_vane[0].railCV(i, j + 1) = Vector3F(*segmentVaneTexcoord(i, 0, j));
				*m_vane[1].railCV(i, j + 1) = Vector3F(*segmentVaneTexcoord(i, 1, j));
			}
		}
	}
	else {
		Vector3F lo(*segmentQuillTexcoord(0));
		Vector3F hi(*segmentQuillTexcoord(1));
		
		hi -= (hi - lo) * .35f;
		Vector3F dv = lo - hi;

		*m_vane[0].railCV(0, 0) = hi;
		*m_vane[0].railCV(1, 0) = hi + dv * .33f;
		*m_vane[0].railCV(2, 0) = hi + dv * .67f;
		*m_vane[0].railCV(3, 0) = lo;
		
		*m_vane[1].railCV(0, 0) = hi;
		*m_vane[1].railCV(1, 0) = hi + dv * .33f;
		*m_vane[1].railCV(2, 0) = hi + dv * .67f;
		*m_vane[1].railCV(3, 0) = lo;
		
		for(short i = 1; i <= numSeg; i++) {	
			*m_vane[0].railCV(0, i) = Vector3F(*segmentQuillTexcoord(i));
			*m_vane[1].railCV(0, i) = Vector3F(*segmentQuillTexcoord(i));
			for(short j = 0; j < 3; j++) {
				*m_vane[0].railCV(j + 1, i) = Vector3F(*segmentVaneTexcoord(i, 0, j));
				*m_vane[1].railCV(j + 1, i) = Vector3F(*segmentVaneTexcoord(i, 1, j));
			}
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
	m_stripe->create(m_resShaft * m_vane[0].gridU() * 2, m_resBarb + 1 + 4);

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
		*m_stripe->currentNumCvs() = nv + 1 + 4;
		
		Vector3F * coord = m_stripe->currentPos();
		Vector3F * col = m_stripe->currentCol();
		m_vane[side].setU(du*i);
		
		for(unsigned j = 0; j < 2; j++) {
            if(!ColorTextureFile.isOpened()) {
                col[j].set(1.f, 1.f, 1.f);
            }
            else {
                m_vane[side].pointOnVane(0.f, coord[j]);
                ColorTextureFile.sample(coord[j].x, coord[j].y, 3, (float *)&col[j]);
            }
        }
			
		for(unsigned j = 0; j <= nv; j++) {
			if(!ColorTextureFile.isOpened()) {
				col[j + 2].set(1.f, 1.f, 1.f);
			}
			else {
				m_vane[side].pointOnVane(dv * j, coord[j + 2]);
				ColorTextureFile.sample(coord[j + 2].x, coord[j + 2].y, 3, (float *)&col[j + 2]);
			}
		}
		
		for(unsigned j = 0; j < 2; j++) {
            if(!ColorTextureFile.isOpened()) {
                col[nv + j + 2].set(1.f, 1.f, 1.f);
            }
            else {
                m_vane[side].pointOnVane(1.f, coord[nv + j + 2]);
                ColorTextureFile.sample(coord[nv + j + 2].x, coord[nv + j + 2].y, 3, (float *)&col[nv + j + 2]);
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

Vector3F * TexturedFeather::patchCenterUV(short seg)
{
	if(type() == 0) return m_vane[0].railCV(seg, 0);
	return m_vane[0].railCV(0, seg);
}

Vector3F * TexturedFeather::patchWingUV(short seg, short side)
{
	if(type() == 0) return m_vane[side].railCV(seg, 3);
	return m_vane[side].railCV(3, seg);
}