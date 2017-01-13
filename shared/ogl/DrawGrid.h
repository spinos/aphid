/*
 *  DrawGrid.h
 *  
 *
 *  Created by jian zhang on 1/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_GRID_H
#define APH_OGL_DRAW_GRID_H

#include <ogl/DrawBox.h>

namespace aphid {

template<typename T>
class DrawGrid : public DrawBox {

	T * m_grid;
	
public:
	DrawGrid(T * g);
	
	void drawCells();
	void drawLevelCells(int level);
	
};

template<typename T>
DrawGrid<T>::DrawGrid(T * g)
{ m_grid = g; }

template<typename T>
void DrawGrid<T>::drawCells()
{
	BoundingBox b;
	m_grid->begin();
	while(!m_grid->end() ) {
		m_grid->getCellBBox(b, m_grid->key() );
		drawBoundingBox(&b);
		
		m_grid->next();
	}
}

template<typename T>
void DrawGrid<T>::drawLevelCells(int level)
{
	BoundingBox b;
	m_grid->begin();
	while(!m_grid->end() ) {
		int l = m_grid->key().w;
		if(l == level) {
			m_grid->getCellBBox(b, m_grid->key() );
			drawBoundingBox(&b);
		}
		if(l > level) {
			return;
		}
		m_grid->next();
	}
}

}
#endif
