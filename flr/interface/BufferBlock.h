/*
 *  BufferBlock.h
 *
 *  a packet of ray and samples
 *  frame with ray origin and direction at tile location
 *  delta ray across frame
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef BUFFER_BLOCK_H
#define BUFFER_BLOCK_H

#include "BufferParameters.h"

class DisplayImage;
class ImageFragment;

class BufferBlock {

/// color
	ImageFragment* m_fragment;
/// frame (origin, direction) 00 10 01 11
/// 00 is left-top corner
/// 11 is right-bottom corner
	float m_frame[4][6];
/// origin, direction, t0, t1
	float m_viewRay[MAX_BUFFER_BLOCK_SIZE][8];
/// normal,depth
	float m_sample[MAX_BUFFER_BLOCK_SIZE][4];
/// average difference after update
	float m_residual;
/// pixel location of tile begin
	int m_tileX, m_tileY;
/// packet dim
	int m_packetX, m_packetY;
	int m_numSamples;
	int m_age;
	
public:
	BufferBlock();
	~BufferBlock();
	
	void setTile(const int& tx, const int& ty);
	//void setEmpty();
	
	const int& tileX() const;
	const int& tileY() const;
	const int& packetX() const;
	const int& packetY() const;
	const int& numSamples() const;
	const int& age() const;
	const float& residual() const;
	
/// i-th frame origin and direction
	void setFrame(int i, const float* ori, const float* dir);
	void setNumSamples(int x);
	void setResidual(const float& x);
	
	void projectImage(DisplayImage* img);
	
	ImageFragment* fragment();
	
/// i-th view ray bilinear interpolate frame by (u,v)
	void calculateViewRay(const float& u, const float& v,
						const int& i);
/// i-th view ray
	const float* viewRay(const int& i) const;
/// age <- 0
/// residual <- random large number
	void begin();
	void progressAge();
	
	static int RoundToBlockSize(int x);
	static int BlockSize();
	
};
#endif