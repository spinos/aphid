/*
 *  BaseBuffer.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif
#include "BaseBuffer.h"

BaseBuffer::BaseBuffer() : _buffereName(0) {}
BaseBuffer::~BaseBuffer() 
{
}

void BaseBuffer::create(float * data, unsigned size)
{
	createVBO(data, size);
}

void BaseBuffer::destroy()
{
	destroyVBO();
}

void BaseBuffer::createVBO(float * data, unsigned size)
{
	if(!data) return;
	
	glGenBuffers(1, &_buffereName);
	
	glBindBuffer(GL_ARRAY_BUFFER, _buffereName);
	
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void BaseBuffer::destroyVBO()
{
	if(_buffereName == 0) return;
	
	glBindBuffer(1, _buffereName);
	glDeleteBuffers(1, &_buffereName);
}

unsigned BaseBuffer::getBufferName() const
{
	return _buffereName;
}