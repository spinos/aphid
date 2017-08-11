/*
 *  DeepBuffer.h
 *  
 *  array of tiled blocks
 *  tile(0,0) is left-top of view port
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DEEP_BUFFER_H
#define DEEP_BUFFER_H

#include "BufferParameters.h"
#include <vector>

class BufferBlock;

class DeepBuffer {

	std::vector<BufferBlock *> m_blocks;
	int m_width, m_height;
	
public:
	DeepBuffer();
	~DeepBuffer();
/// add blocks if necessary	
	void create(int w, int h);
	
	const int& width() const;
	const int& height() const;
/// width x height
	int bufferSize() const;
	int numBlockX() const;
	int numBlockY() const;
	int numBlocks() const;
/// i-th block
	BufferBlock* block(int i);
	
private:
	void addBlocks(int nblk);
	void initBlocks();
	
};

#endif
