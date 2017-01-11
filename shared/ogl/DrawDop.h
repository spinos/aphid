/*
 *  DrawDop.h
 *  
 *	discrete oriented polytope
 *
 *  Created by jian zhang on 1/11/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_DOP_H
#define APH_OGL_DRAW_DOP_H

namespace aphid {

class AOrientedBox;

class DrawDop {

	float * m_vertexNormals;
	float * m_vertexPoints;
	int m_numVertices;
	
public:

	DrawDop();
	virtual ~DrawDop();
	
	void update8DopPoints(const AOrientedBox & ob);
	
	void drawAWireDop() const;
	void drawASolidDop() const;

protected:

private:
	void clear();
	
};

}
#endif