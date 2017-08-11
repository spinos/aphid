/*
 *  DisplayImage.h
 *  
 *  image content, frame origin pixel
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DISPLAY_IMAGE_H
#define DISPLAY_IMAGE_H

typedef unsigned char uchar;

class DisplayImage {

	uchar* m_imageBits[2];
	int m_curBits;
	int m_xres, m_yres;
	int m_capacity;
	
public:
	DisplayImage();
	~DisplayImage();
	
	void create(int w, int h);
	
	const int& xres() const;
	const int& yres() const;
	
	const uchar* bits() const;
	uchar* scanline(int i);
	
private:
	void createSwapBits(int w, int h);
	
};

#endif
