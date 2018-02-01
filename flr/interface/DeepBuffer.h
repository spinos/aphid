/*
 *  DeepBuffer.h
 *  
 *  array of tiled blocks
 *  tile(0,0) is left-top of view port
 *  priority map to each block
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DEEP_BUFFER_H
#define DEEP_BUFFER_H

#include "BufferParameters.h"
#include <boost/scoped_array.hpp>
#include <vector>

namespace aphid {
template <typename KeyType, typename ValueType>
struct QuickSortPair;
}

class BufferBlock;
class DisplayCamera;

class DeepBuffer {

	std::vector<BufferBlock *> m_blocks;
	int m_width, m_height;
	boost::scoped_array<aphid::QuickSortPair<float, int> > m_priority;
	
public:
	DeepBuffer();
	~DeepBuffer();
/// add blocks if necessary	
	void create(int w, int h);
/// for each block update frame and begin
	void setBegin(const DisplayCamera* camera);
	
	const int& width() const;
	const int& height() const;
/// width x height
	int bufferSize() const;
	int numBlockX() const;
	int numBlockY() const;
	int numBlocks() const;
/// i-th block
	BufferBlock* block(int i);
	BufferBlock* highResidualBlock();
/// max residual of all blocks
	float maxResidual() const;
	
private:
	void addBlocks(int nblk);
	void initBlocks();
/// first one has residual > thre in priority
/// skip block has residual < thre
	int findPriorityBegin(const float& thre) const;
	
};

#endif
