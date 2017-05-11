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
	float * m_vertexColors;
	int m_numVertices;
	
public:

	DrawDop();
	virtual ~DrawDop();
	
	void update8DopPoints(const AOrientedBox & ob,
	                    const float * sizing=0 );
	
	void drawAWireDop() const;
	void drawASolidDop() const;
	
	const int & dopBufLength() const;
	const float * dopPositionBuf() const;
	const float * dopNormalBuf() const;
	const float * dopColorBuf() const;

protected:
	void setUniformDopColor(const float * c);
	
private:
	void clear();
	
};

}
#endif