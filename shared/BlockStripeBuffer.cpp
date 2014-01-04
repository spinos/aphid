/*
 *  BlockStripeBuffer.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/4/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "BlockStripeBuffer.h"
#include "AdaptableStripeBuffer.h"
BlockStripeBuffer::BlockStripeBuffer() {}
BlockStripeBuffer::~BlockStripeBuffer() { clear(); }

void BlockStripeBuffer::clear() 
{
	for (std::vector<AdaptableStripeBuffer *>::iterator it = m_blocks.begin(); 
				it != m_blocks.end(); ++it)
		delete *it;
	m_blocks.clear();
}

void BlockStripeBuffer::initialize()
{
	AdaptableStripeBuffer * b = new AdaptableStripeBuffer;
	b->create(524288);
	m_blocks.push_back(b);
	begin();
}

AdaptableStripeBuffer * BlockStripeBuffer::currentBlock()
{
	return m_blocks[m_currentBlockIdx];
}

AdaptableStripeBuffer * BlockStripeBuffer::nextBlock()
{
	m_currentBlockIdx++;
	if(numBlocks() == m_currentBlockIdx) {
		AdaptableStripeBuffer * b = new AdaptableStripeBuffer;
		b->create(524288);
		m_blocks.push_back(b);
	}
	m_blocks[m_currentBlockIdx]->begin();
	return m_blocks[m_currentBlockIdx];
}

void BlockStripeBuffer::append(AdaptableStripeBuffer * src)
{
	const unsigned ns = src->numStripe();
	unsigned * ncv = src->numCvs();
	Vector3F * pos = src->pos();
	Vector3F * col = src->col();
	float * w = src->width();
	unsigned ncvi;
	AdaptableStripeBuffer * dst = currentBlock();
	
	for(unsigned i = 0; i < ns; i++) {
		ncvi = ncv[i];
		if(!dst->canContain(ncvi)) dst = nextBlock();
		*dst->currentNumCvs() = ncvi;
		for(unsigned j = 0; j < ncvi; j++) {
			dst->currentPos()[j] = pos[j];
			dst->currentCol()[j] = col[j];
			dst->currentWidth()[j] = w[j];
		}
		pos += ncvi;
		col += ncvi;
		w += ncvi;
		
		dst->next();
	}
}

unsigned BlockStripeBuffer::numBlocks() const
{
	return m_blocks.size();
}

AdaptableStripeBuffer * BlockStripeBuffer::block(unsigned idx) const
{
	return m_blocks[idx];
}

void BlockStripeBuffer::begin()
{
	m_currentBlockIdx = 0;
	m_blocks[0]->begin();
}
