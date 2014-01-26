/*
 *  PatchTexture.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "PatchTexture.h"
#include "PatchMesh.h"

PatchTexture::PatchTexture() 
{ 
	setTextureFormat(FFloat);
	setTextureDepth(DThree);
}

PatchTexture::~PatchTexture() { m_colors.reset(); }

int PatchTexture::resolution() const { return 4; }

void PatchTexture::create(PatchMesh * mesh)
{
	const int ppp = 5 * 5;
	const unsigned nf = mesh->numQuads();
	m_numColors = nf * ppp;
	m_colors.reset(new Float3[m_numColors]);
	for(unsigned i = 0; i < m_numColors; i++) m_colors[i].set(1.f, 1.f, 1.f);
	setAllWhite(true);
}

unsigned PatchTexture::numTexels() const { return m_numColors; }

void * PatchTexture::data() { return m_colors.get(); }

void * PatchTexture::data() const { return m_colors.get(); }

Float3 * PatchTexture::patchColor(const unsigned & idx) { return &m_colors[25 * idx]; }

void PatchTexture::sample(const unsigned & faceIdx, const float & faceU, const float & faceV, Float3 & dst) const
{
	Float3 * pix = &m_colors[25 * faceIdx];
	const int gu0 = faceU * 4.f;
	const int gv0 = faceV * 4.f;
	int gu1 = gu0 + 1;
	int gv1 = gv0 + 1;
	if(gu1 > 4) gu1 = 4;
	if(gv1 > 4) gv1 = 4;
	const float pu = faceU * 4.f - gu0;
	const float pv = faceV * 4.f - gv0;
	
	Float3 a = pix[5 * gv0 + gu0];
	Float3 b = pix[5 * gv0 + gu1];
	Float3 c = pix[5 * gv1 + gu0];
	Float3 d = pix[5 * gv1 + gu1];
	
	a.set(a.x * (1.f - pu) + b.x * pu, a.y * (1.f - pu) + b.y * pu, a.z * (1.f - pu) + b.z * pu);
	c.set(c.x * (1.f - pu) + d.x * pu, c.y * (1.f - pu) + d.y * pu, c.z * (1.f - pu) + d.z * pu);
	
	dst.set(a.x * (1.f - pv) + c.x * pv, a.y * (1.f - pv) + c.y * pv, a.z * (1.f - pv) + c.z * pv);
}

void PatchTexture::fillPatchColor(const unsigned & faceIdx, const Float3 & fillColor)
{
	Float3 * pix = &m_colors[25 * faceIdx];
	for(unsigned i = 0; i < 25; i++) pix[i] = fillColor;
}

void PatchTexture::toUChar(const Float3 & src, unsigned char * dst) const
{
	dst[0] = 255 * src.x;
	dst[1] = 255 * src.y;
	dst[2] = 255 * src.z;
}

void PatchTexture::toFloat(unsigned char * src, Float3 & dst) const
{
	dst.x = ((float)((int)src[0])) / 255.f;
	dst.y = ((float)((int)src[1])) / 255.f;
	dst.z = ((float)((int)src[2])) / 255.f;
}
