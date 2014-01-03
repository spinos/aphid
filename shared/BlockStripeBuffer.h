/*
 *  BlockStripeBuffer.h
 *  aphid
 *
 *  Created by jian zhang on 1/4/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <AllMath.h>
class AdaptableStripeBuffer;
class BlockStripeBuffer {
public:
	BlockStripeBuffer();
	virtual ~BlockStripeBuffer();
	
	void initialize();
	void append(AdaptableStripeBuffer * buffer);
	
	unsigned numBlocks() const;
	AdaptableStripeBuffer * block(unsigned idx) const;
	
	void begin();
protected:
	AdaptableStripeBuffer * currentBlock();
	AdaptableStripeBuffer * nextBlock();
private:
	void clear();
private:
	std::vector<AdaptableStripeBuffer *> m_blocks;
	unsigned m_currentBlockIdx;
};