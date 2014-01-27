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
#include <ColorBlend.h>

TexturePainter::TexturePainter() 
{ 
	m_blend = new ColorBlend; 
	m_mode = MReplace;
	m_lastPosition.set(-10e8, -10e8, -10e8);
	m_averageFaceSize = 0.f;
}

TexturePainter::~TexturePainter() { delete m_blend; }

void TexturePainter::setBrush(BaseBrush * brush) { m_brush = brush; }

char TexturePainter::updatePaintPosition()
{
	if(m_lastPosition.distanceTo(m_brush->heelPosition()) < m_averageFaceSize) return 0;
	m_lastPosition = m_brush->heelPosition();
	return 1;
}

void TexturePainter::paintOnMeshFaces(PatchMesh * mesh, const std::deque<unsigned> & faceIds, BaseTexture * tex)
{
	if(!updatePaintPosition()) return;
	
    if(m_mode == MReplace) m_destinyColor = m_brush->color();
    else m_destinyColor = averageColor(faceIds, tex);

	PatchTexture * ptex = static_cast<PatchTexture *>(tex);
	
	m_averageFaceSize = 0.f;
	int res = ptex->resolution();
	std::deque<unsigned>::const_iterator it = faceIds.begin();
	for(; it != faceIds.end(); ++it) {
		Patch p = mesh->patchAt(*it);
		paintOnFace(p, ptex->patchColor(*it), res);
		m_averageFaceSize += p.size();
	}
	m_averageFaceSize /= (float)faceIds.size();
	m_averageFaceSize *= .33f;
	if(Vector3F((float *)&m_destinyColor) != Vector3F::One) tex->setAllWhite(false);
}

void TexturePainter::paintOnFace(const Patch & face, Float3 * tex, const int & ngrid)
{
	m_blend->setCenter(m_brush->heelPosition());
	m_blend->setMaxDistance(m_brush->radius());
	m_blend->setDropoff(m_brush->dropoff());
	m_blend->setStrength(m_brush->strength());
	
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
			
			m_blend->blend(pop, m_destinyColor, &tex[acc++]);
		}
	}
}

void TexturePainter::setPaintMode(TexturePainter::PaintMode m) { m_mode = m; }
TexturePainter::PaintMode TexturePainter::paintMode() const { return m_mode; }

Float3 TexturePainter::averageColor(const std::deque<unsigned> & faceIds, BaseTexture * tex) const
{
    Vector3F sum;
    Float3 sample;
    float u, v;
    int j, nsamp = 0;
    std::deque<unsigned>::const_iterator it = faceIds.begin();
	for(; it != faceIds.end(); ++it) {
	    for(j = 0; j < 4; j++) {
	        u = (float)(rand() % 199) / 199.f;
	        v = (float)(rand() % 199) / 199.f;
	        tex->sample(*it, u, v, sample);
	        sum += Vector3F((float *)&sample);
	        nsamp++;
	    }
	}
	sum /= (float)nsamp;
	return Float3(sum.x, sum.y, sum.z);
}
