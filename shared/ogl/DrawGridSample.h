/*
 *  DrawGridSample.h
 *  
 *  samples stored in grid cells
 *  T as grid type, Tc as cell type, Tn as node type
 *
 *  Created by jian zhang on 1/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_GRID_SAMPLE_H
#define APH_OGL_DRAW_GRID_SAMPLE_H

#include <ogl/DrawGlyph.h>
#include <gl_heads.h>

namespace aphid {

template<typename T, typename Tc, typename Tn>
class DrawGridSample {

	GlslLegacyInstancer * m_instancer;
    T * m_grid;
	
public:
	DrawGridSample(T * g);
	
	bool initGlsl();
	
	void drawLevelSamples(int level);

private:
	void drawSamplesInCell(Tc * cell, Float4 * row);
	
};

template<typename T, typename Tc, typename Tn>
DrawGridSample<T, Tc, Tn>::DrawGridSample(T * g)
{ 
	m_instancer = new GlslLegacyInstancer; 
    m_grid = g; 
}

template<typename T, typename Tc, typename Tn>
bool DrawGridSample<T, Tc, Tn>::initGlsl()
{
	std::string diaglog;
    m_instancer->diagnose(diaglog);
    std::cout<<diaglog;
    m_instancer->initializeShaders(diaglog);
    std::cout<<diaglog;
    std::cout.flush();
    return m_instancer->isDiagnosed();
}

template<typename T, typename Tc, typename Tn>
void DrawGridSample<T, Tc, Tn>::drawLevelSamples(int level)
{
	const float sz = m_grid->levelCellSize(level + 2);
	Float4 row[4];
    row[0].set(sz,0,0,0);
    row[1].set(0,sz,0,0);
    row[2].set(0,0,sz,0);
    row[3].set(1,1,0,0);
	
	glBegin(GL_POINTS);
	
	m_grid->begin();
	while(!m_grid->end() ) {
		int l = m_grid->key().w;
		if(l == level) {
			
			drawSamplesInCell(m_grid->value(), row );
			
		}
		if(l > level) {
			return;
		}
		m_grid->next();
	}
	
	glEnd();
	
}

template<typename T, typename Tc, typename Tn>
void DrawGridSample<T, Tc, Tn>::drawSamplesInCell(Tc * cell, Float4 * row)
{
	cell->begin();
	while (!cell->end() ) {
		Tn * nd = cell->value();
		
	    glVertex3fv((const float *)&nd->pos);
		
		cell->next();
	}
}

}
#endif
