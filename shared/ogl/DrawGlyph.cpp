/*
 *  DrawGlyph.cpp
 *  
 *
 *  Created by jian zhang on 1/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawGlyph.h"
#include <geom/SuperQuadricGlyph.h>
#include <gl_heads.h>

namespace aphid {

DrawGlyph::DrawGlyph()
{
	m_glyph = new SuperQuadricGlyph(8);
	m_glyph->computePositions(1.f, 1.f);	
}
	
DrawGlyph::~DrawGlyph()
{
	delete m_glyph;
}

void DrawGlyph::updateGlyph(float a, float b)
{
	m_glyph->computePositions(a, b);
}

void DrawGlyph::drawAGlyph() const
{
	glNormalPointer(GL_FLOAT, 0, (const GLfloat*)m_glyph->vertexNormals());
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)m_glyph->points());
	glDrawElements(GL_TRIANGLES, m_glyph->numIndices(), GL_UNSIGNED_INT, m_glyph->indices());
}

}
