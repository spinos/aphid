/*
 *  DrawGlyph.h
 *  
 *
 *  Created by jian zhang on 1/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_GLYPH_H
#define APH_OGL_DRAW_GLYPH_H

namespace aphid {

class SuperQuadricGlyph;

class DrawGlyph {

	SuperQuadricGlyph * m_glyph;
	
public:
	DrawGlyph();
	virtual ~DrawGlyph();
	
protected:
	void updateGlyph(float a, float b);
	
	void drawAGlyph() const;
	
};

}
#endif
