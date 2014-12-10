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

BaseBuffer::BaseBuffer() : m_bufferName(0), m_bufferSize(0) 
{
	m_bufferType = kUnknown;
}

BaseBuffer::~BaseBuffer() 
{
}

void BaseBuffer::create(float * data, unsigned size)
{
	m_bufferSize = size;
	createVBO(data, size);
	setBufferType(kVBO);
}

void BaseBuffer::destroy()
{
	destroyVBO();
}

void BaseBuffer::createVBO(float * data, unsigned size)
{
	glGenBuffers(1, &m_bufferName);
	
	glBindBuffer(GL_ARRAY_BUFFER, m_bufferName);
	
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void BaseBuffer::destroyVBO()
{
	if(m_bufferName == 0) return;
	
	glBindBuffer(1, m_bufferName);
	glDeleteBuffers(1, &m_bufferName);
}

const unsigned BaseBuffer::bufferName() const { return m_bufferName; }

const unsigned BaseBuffer::bufferSize() const { return m_bufferSize; }

void BaseBuffer::setBufferType(BaseBuffer::BufferType t) { m_bufferType = t; }
const BaseBuffer::BufferType BaseBuffer::bufferType() const { return m_bufferType; }