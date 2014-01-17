/*
 *  BlockBlockDrawBuffer.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BlockDrawBuffer.h"

#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif
BlockDrawBuffer::BlockDrawBuffer() 
{
    initializeBuffer();
}

BlockDrawBuffer::~BlockDrawBuffer() 
{
    clearBuffer();
}

void BlockDrawBuffer::initializeBuffer()
{
	if(numBlock() > 0) clearBuffer();
	m_blocks.push_back(new PtrTup);
	begin();
	m_taken = 0;
}

void BlockDrawBuffer::clearBuffer()
{
    for (std::vector<PtrTup *>::iterator it = m_blocks.begin(); 
				it != m_blocks.end(); ++it)
		delete *it;
	m_blocks.clear();
	m_current = 0;
}

unsigned BlockDrawBuffer::capacity() const 
{
	return numBlock() * numElementPerBlock();
}

unsigned BlockDrawBuffer::numElementPerBlock() const
{
	return 32768;
}

unsigned BlockDrawBuffer::numBlock() const
{
	return m_blocks.size();
}

void BlockDrawBuffer::expandBy(const unsigned & size)
{
	if(m_current + size >= capacity()) {
		const unsigned blockToCreate = (m_current + size) / numElementPerBlock() + 1 - numBlock();
		for(unsigned i = 0; i < blockToCreate; i++) {
			PtrTup * t = new PtrTup();
			m_blocks.push_back(t);
		}
	}
	if(m_current + size > m_taken)
		m_taken = m_current + size;
}

void BlockDrawBuffer::begin()
{
	m_current = 0;
	m_vertexPtr = m_blocks[0]->alignedV;
	m_normalPtr = m_blocks[0]->alignedN;
	m_texcoordPtr = m_blocks[0]->alignedT;
}

void BlockDrawBuffer::next()
{
	m_current++;
	if(end()) return;
	if(m_current % numElementPerBlock() == 0) {
		unsigned blockIdx = m_current / numElementPerBlock();
		m_vertexPtr = m_blocks[blockIdx]->alignedV;
		m_normalPtr = m_blocks[blockIdx]->alignedN;
		m_texcoordPtr = m_blocks[blockIdx]->alignedT;
	}
	else {
		m_vertexPtr += 12;
		m_normalPtr += 12;
		m_texcoordPtr += 8;
	}
}

char BlockDrawBuffer::end() const
{
	return m_current >= capacity();
}

void BlockDrawBuffer::setIndex(unsigned index)
{
	m_current = index;
	unsigned blockIdx = m_current / numElementPerBlock();
	unsigned offset = m_current % numElementPerBlock();
	m_vertexPtr = m_blocks[blockIdx]->alignedV + offset * 12;
	m_normalPtr = m_blocks[blockIdx]->alignedN + offset * 12;
	m_texcoordPtr = m_blocks[blockIdx]->alignedT + offset * 8;
}

void BlockDrawBuffer::drawBuffer() const
{
	for(unsigned i = 0; i < numBlock(); i++) {
		glEnableClientState( GL_VERTEX_ARRAY );
		glVertexPointer( 3, GL_FLOAT, 0, (float *)m_blocks[i]->alignedV);
	
		glEnableClientState( GL_NORMAL_ARRAY );
		glNormalPointer( GL_FLOAT, 0, (float *)m_blocks[i]->alignedN );
		
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(2, GL_FLOAT, 0, (float *)m_blocks[i]->alignedT);
	
		if(i == numBlock() - 1)
			glDrawArrays( GL_QUADS, 0, m_taken % numElementPerBlock());
		else 
			glDrawArrays( GL_QUADS, 0, numElementPerBlock());
	
		glDisableClientState( GL_NORMAL_ARRAY );
		glDisableClientState( GL_VERTEX_ARRAY );
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);		
	}
}

float * BlockDrawBuffer::vertices()
{
	return (float *)m_vertexPtr;
}

float * BlockDrawBuffer::normals()
{
	return (float *)m_normalPtr;
}

float * BlockDrawBuffer::texcoords()
{
	return (float *)m_texcoordPtr;
}

unsigned BlockDrawBuffer::taken() const
{
	return m_taken;
}

unsigned BlockDrawBuffer::index() const
{
	return m_current;
}
