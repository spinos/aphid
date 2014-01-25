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

PatchTexture::PatchTexture() {}
PatchTexture::~PatchTexture() {}

void PatchTexture::create(PatchMesh * mesh)
{
	const int ppp = 5 * 5;
	const unsigned nf = mesh->numQuads();
	m_numColors = nf * ppp;
	m_colors.reset(new Float3[m_numColors]);
	for(unsigned i = 0; i < m_numColors; i++) m_colors[i].set(.75f, .75f, .75f);
}

const unsigned & PatchTexture::numColors() const { return m_numColors; }

Float3 * PatchTexture::colors() { return m_colors.get(); }

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

