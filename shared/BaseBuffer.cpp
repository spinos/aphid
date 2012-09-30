/*
 *  BaseBuffer.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#include "BaseBuffer.h"

BaseBuffer::BaseBuffer() : _buffereName(0) {}
BaseBuffer::~BaseBuffer() 
{
	destroy();
}

void BaseBuffer::create(float * data, unsigned size)
{
	if(!data) return;
	
	glGenBuffers(1, &_buffereName);
	
	glBindBuffer(GL_ARRAY_BUFFER, _buffereName);
	
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);	
}

void BaseBuffer::destroy()
{
	if(_buffereName == 0) return;
	
	glBindBuffer(1, _buffereName);
	glDeleteBuffers(1, &_buffereName);
}

unsigned BaseBuffer::getBufferName() const
{
	return _buffereName;
}