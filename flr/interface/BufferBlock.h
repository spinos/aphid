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

class BufferBlock {

/// frame 00 10 01 11
	float m_frameOrigin[4][3];
	float m_frameDirection[4][3];
/// ray
	float m_origin[MAX_BUFFER_BLOCK_SIZE][3];
	float m_direction[MAX_BUFFER_BLOCK_SIZE][3];
/// sample
	float m_color[MAX_BUFFER_BLOCK_SIZE][4];
	float m_position[MAX_BUFFER_BLOCK_SIZE][3];
/// pixel location of tile begin
	int m_tileX, m_tileY;
/// packet dim
	int m_packetX, m_packetY;
	int m_numSamples;
	
public:
	BufferBlock();
	~BufferBlock();
	
	void setTile(const int& tx, const int& ty);
	void setEmpty();
	
	const int& tileX() const;
	const int& tileY() const;
	const int& numSamples() const;
	
	void setFrame(int i, const float* ori, const float* dir);
	void setNumSamples(int x);
	
	void projectImage(DisplayImage* img);
	
/// i-th sample
	float* sampleColor(int i);
	
	static int RoundToBlockSize(int x);
	static int BlockSize();
	
};
#endif