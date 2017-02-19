/*
 *  DrawSample.h
 *  
 *  as points
 *
 *  Created by jian zhang on 1/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_SAMPLE_H
#define APH_OGL_DRAW_SAMPLE_H

namespace aphid {

class DrawSample {
	
public:
	struct Profile {
		int m_stride;
		float m_pointSize;
		bool m_hasNormal;
		bool m_hasColor;
	};
	
private:
	Profile m_prof;
	
public:
	DrawSample();
	
	void begin(const Profile & prof);
	void end() const;
	void drawColored(const float * points,
				const float * colors,
				const int & count) const;
	void draw(const float * points,
				const float * normals,
				const int & count) const;
	void draw(const float * points,
				const float * normals,
				const int * indices,
				const int & count) const;
	void drawColored(const float * points,
				const float * colors,
				const int * indices,
				const int & count) const;
	
private:
	
};

}
#endif
