/*
 *  TexturePainter.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "TexturePainter.h"
#include "BaseBrush.h"
#include "PatchMesh.h"
#include "PatchTexture.h"
#include "PointInsidePolygonTest.h"
TexturePainter::TexturePainter() {}
TexturePainter::~TexturePainter() {}

void TexturePainter::setBrush(BaseBrush * brush) { m_brush = brush; }

void TexturePainter::paintOnMeshFaces(PatchMesh * mesh, const std::deque<unsigned> & faceIds, BaseTexture * tex)
{
	PatchTexture * ptex = static_cast<PatchTexture *>(tex);
		
	int res = ptex->resolution();
	std::deque<unsigned>::const_iterator it = faceIds.begin();
	for(; it != faceIds.end(); ++it) {
		Patch p = mesh->patchAt(*it);
		paintOnFace(p, ptex->patchColor(*it), res);
	}
}

void TexturePainter::paintOnFace(const Patch & face, Float3 * tex, const int & ngrid)
{
	m_blend.setCenter(m_brush->heelPosition());
	m_blend.setMaxDistance(m_brush->radius());
	const Float3 dstCol = m_brush->color();
	Vector3F pop;
	const float du = 1.f / (float)ngrid;
	const float dv = du;
	float u, v;
	int acc = 0;
	for(int j = 0; j <= ngrid; j++) {
		v = dv * j;
		for(int i = 0; i <= ngrid; i++) {
			u = du * i;
			face.point(u, v, &pop);
			
			m_blend.blend(pop, dstCol, &tex[acc++]);
		}
	}
}
