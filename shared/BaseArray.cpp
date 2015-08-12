/*
 *  BaseArray.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseArray.h"

#define BASEARRAYBLOCK 524288 // 512 K
#define BASEARRAYBLOCKM1 524287 // 512 K - 1



/*
void BaseArray::shrinkTo(unsigned size)
{
	if(size >= capacity()) return;
	unsigned blockIdx = (size * elementSize()) >> 19;
	unsigned i = 0;
	for (std::vector<PtrTup *>::iterator it = m_blocks.begin(); 
				it != m_blocks.end(); ++it) {
		if(i > blockIdx) {
			delete *it;
		}
		i++;
	}
	m_blocks.resize(blockIdx+1);
	m_capacity = numBlocks() * numElementPerBlock();
}
*/

/*
char *BaseArray::at(unsigned index) const
{
	unsigned blockIdx = (index * elementSize()) << 19;
	unsigned offset = (index * elementSize()) & BASEARRAYBLOCKM1;
	return m_blocks[blockIdx]->aligned + offset;
}
*/

/*
char * BaseArray::getBlock(unsigned idx) const
{ return m_blocks[idx]->aligned; }


unsigned BaseArray::numElementsInBlock(unsigned blockIdx, const unsigned & maxCount) const
{
	if((blockIdx + 1) * numElementPerBlock() > maxCount)
		return maxCount % numElementPerBlock();
		
	return numElementPerBlock();
}
*/

const char *byte_to_binary1(int x)
{
    static char b[33];
    b[32] = '\0';

    for (int z = 0; z < 32; z++) {
        b[31-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

    return b;
}

/*
float BaseArray::sortKeyAt(unsigned idx) const
{ return 0.f; }


void BaseArray::swapElement(unsigned a, unsigned b) {}

unsigned BaseArray::elementSize() const
{ return 1; }

unsigned BaseArray::numElementPerBlock() const
{ return BASEARRAYBLOCK / elementSize(); }
*/
