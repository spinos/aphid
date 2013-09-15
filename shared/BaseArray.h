/*
 *  BaseArray.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <vector>
class BaseArray {
public:
	BaseArray();
	virtual ~BaseArray();

	void clear();
	
	char *expandBy(unsigned size);
	void shrinkTo(unsigned size);
	
	void begin();
	void next();
	char end() const;
	
	unsigned index() const;
	void setIndex(unsigned index);
	
	char *current();
	char *at(unsigned index);
	char *at(unsigned index) const;
	
	unsigned capacity() const;
	unsigned numElementPerBlock() const;
	unsigned numBlock() const;
	
	void setElementSize(unsigned size);
	unsigned getElementSize() const;
	
	virtual float sortKeyAt(unsigned idx) const;
	virtual void swapElement(unsigned a, unsigned b);
	
	void verbose() const;
	
private:
	struct PtrTup {
		PtrTup() {
			raw = new char[524288 + 31];
			aligned = (char *)(((unsigned long)raw + 32) & (0xffffffff - 31));
		}
		~PtrTup() {
			delete[] raw;
		}
		
		char *raw;
		char *aligned;
	};
	
	std::vector<PtrTup *> m_blocks;
	unsigned m_elementSize;
	unsigned m_current;
	char *m_ptr;
};