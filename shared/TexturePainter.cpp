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

void TexturePainter::bufferFaces(PatchMesh * mesh, const std::deque<unsigned> & faceIds)
{
	m_faces.reset(new Patch[faceIds.size()]);
	std::deque<unsigned>::const_iterator it = faceIds.begin();
	for(unsigned i = 0; it != faceIds.end(); ++it, ++i) {
		m_faces[i] = mesh->patchAt(*it);
	}
}

void TexturePainter::paintOnMeshFaces(PatchMesh * mesh, const std::deque<unsigned> & faceIds, BaseTexture * tex)
{
	if(!updatePaintPosition()) return;
	
	bufferFaces(mesh, faceIds);
	
    if(m_mode == MReplace) {
		m_destinyColor = m_brush->color();
		m_blend->setDropoff(m_brush->dropoff());
		m_blend->setStrength(m_brush->strength());
	}
    else {
		m_destinyColor = averageColor(faceIds, tex);
		m_blend->setDropoff(1.f);
		m_blend->setStrength(.25f);
	}
	
	m_blend->setCenter(m_brush->heelPosition());
	m_blend->setMaxDistance(m_brush->radius());

	PatchTexture * ptex = static_cast<PatchTexture *>(tex);
	
	m_averageFaceSize = 0.f;
	int res = ptex->resolution();
	std::deque<unsigned>::const_iterator it = faceIds.begin();
	for(unsigned i = 0; it != faceIds.end(); ++it, ++i) {
		paintOnFace(m_faces[i], ptex->patchColor(*it), res);
		m_averageFaceSize += m_faces[i].size();
	}
	m_averageFaceSize /= (float)faceIds.size();
	m_averageFaceSize *= .33f;
	if(Vector3F((float *)&m_destinyColor) != Vector3F::One) tex->setAllWhite(false);
}

void TexturePainter::paintOnFace(const Patch & face, Float3 * tex, const int & ngrid)
{
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
    Vector3F sum, pop;
    Float3 sample;
    float u, v, d, weight, nsamp = 0.f;
    int j;
    std::deque<unsigned>::const_iterator it = faceIds.begin();
	for(unsigned i = 0; it != faceIds.end(); ++it, ++i) {
		Patch & p = m_faces[i];
	    for(j = 0; j < 4; j++) {
	        u = (float)(rand() % 199) / 199.f;
	        v = (float)(rand() % 199) / 199.f;
			p.point(u, v, &pop);
			
			d = m_lastPosition.distanceTo(pop);
			if(d >= m_brush->radius()) continue;
			
			weight = 1.f - d / m_brush->radius();
			
	        tex->sample(*it, u, v, sample);
	        sum += Vector3F((float *)&sample) * weight;
	        nsamp += weight;
	    }
	}
	if(nsamp < 10e-5) return Float3(0.f, 0.f, 0.f);
	sum /= nsamp;
	return Float3(sum.x, sum.y, sum.z);
}
