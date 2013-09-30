/*
 *  BlockDrawBuffer.h
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <AllMath.h>

class BlockDrawBuffer {
public:
    BlockDrawBuffer();
    virtual ~BlockDrawBuffer();
    
    void initializeBuffer();
    void clearBuffer();
    void drawBuffer() const;
	
	unsigned capacity() const;
	unsigned numElementPerBlock() const;
	unsigned numBlock() const;
	
	void expandBy(unsigned size);
	
	void begin();
	void next();
	char end() const;
	
	void setIndex(unsigned index);
	unsigned index() const;
	unsigned taken() const;
    
	float * vertices();
	float * normals();
private:
	struct PtrTup {
		PtrTup() {
			rawV = new char[32768 * 12 + 31];
			alignedV = (char *)(((unsigned long)rawV + 32) & (0xffffffff - 31));
			rawN = new char[32768 * 12 + 31];
			alignedN = (char *)(((unsigned long)rawN + 32) & (0xffffffff - 31));
		}
		~PtrTup() {
			delete[] rawV;
			delete[] rawN;
		}
		
		char *rawV;
		char *alignedV;
		char *rawN;
		char *alignedN;
	};
	
	std::vector<PtrTup *> m_blocks;
	unsigned m_current, m_taken;
	char * m_vertexPtr;
	char * m_normalPtr;
};
