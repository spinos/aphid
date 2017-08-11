/*
 *  DisplayImage.cpp
 *  
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DisplayImage.h"
#include <iostream>

DisplayImage::DisplayImage() :
m_curBits(0),
m_xres(0),
m_yres(0),
m_capacity(0)
{
	m_imageBits[0] = 0;
	m_imageBits[1] = 0;
}

DisplayImage::~DisplayImage()
{}

void DisplayImage::create(int w, int h)
{
	if(m_capacity < w * h) {
		createSwapBits(w, h);
	}
	m_xres = w;
	m_yres = h;
}

void DisplayImage::createSwapBits(int w, int h)
{
	m_capacity = w * h;
	std::cout<<" buf xy "<<w<<" "<<h;
	m_curBits++;
	if(m_curBits>1) m_curBits = 0;
	
	int a = m_curBits;
	m_imageBits[a] = new uchar[m_capacity * 4];
	std::cout<<" crt "<<a;
	
	int b = m_curBits+1;
	if(b>1) b = 0;
	if(m_imageBits[b]) {
		delete[] m_imageBits[b];
		std::cout<<" cln "<<b;
	}
	
	std::cout.flush();
}

const int& DisplayImage::xres() const
{ return m_xres; }

const int& DisplayImage::yres() const
{ return m_yres; }

const uchar* DisplayImage::bits() const
{ return m_imageBits[m_curBits]; }

uchar* DisplayImage::scanline(int i)
{ return &m_imageBits[m_curBits][i * m_xres * 4]; }
	