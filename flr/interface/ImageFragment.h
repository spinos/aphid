/*
 *  ImageFragment.h
 *  
 *  a piece of the image	
 *
 *  Created by jian zhang on 2/5/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef IMAGE_FRAGMENT_H
#define IMAGE_FRAGMENT_H

#include "BufferParameters.h"

class ImageFragment {

/// pixel rgba
	float m_color[4][MAX_BUFFER_BLOCK_SIZE];
/// size
	int m_fragWidth, m_fragHeight;
	
public:
	ImageFragment();
/// i-th pixel
	void setColor(const float* x, const int& i);
	
	const int& fragmentWidth() const;
	const int& fragmentHeight() const;
	
	float* colorComponent(int i);
	const float* colorComponent(int i) const;
	
private:

};

#endif
