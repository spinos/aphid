/*
 *  DrawHeightField.h
 *  
 *
 *  Created by jian zhang on 3/23/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef WBG_DRAW_HEIGHT_FIELD_H
#define WBG_DRAW_HEIGHT_FIELD_H

#include <boost/scoped_array.hpp>

namespace aphid {

class Ray;
class Vector2F;

namespace img {

class HeightField;

}

}

class DrawHeightField {

	boost::scoped_array<float> m_pos;
	boost::scoped_array<float> m_col;
	int m_numVertices;
	int m_curFieldInd;
	float m_planeHeight;

public:
	DrawHeightField();
	virtual ~DrawHeightField();
	
	void bufferValue(const aphid::img::HeightField & fld);
	
	void drawBound(const aphid::img::HeightField & fld) const;
	void drawValue(const aphid::img::HeightField & fld) const;
	
	void setCurFieldInd(int x);
	const int & curFieldInd() const;
	
	const float & planeHeight() const;
	
protected:

private:
};

#endif