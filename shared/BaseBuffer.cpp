/*
 *  BaseBuffer.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "BaseBuffer.h"
#include <iostream>
BaseBuffer::BaseBuffer() : m_bufferSize(0), m_native(0)
{}

BaseBuffer::~BaseBuffer() 
{ destroy(); }

void BaseBuffer::create(unsigned size)
{
	if(size <= m_bufferSize) {
		m_bufferSize = size;
		return;
	}
	
	destroy();
    m_native = new char[size];
    m_bufferSize = size;
}

void BaseBuffer::destroy()
{ if(m_native) delete[] m_native; }

const unsigned BaseBuffer::bufferSize() const 
{ return m_bufferSize; }

char * BaseBuffer::data() const 
{return m_native; }

void BaseBuffer::copyFrom(const void * src, unsigned size)
{ memcpy( data(), src, size ); }

void BaseBuffer::copyFrom(const void * src, unsigned size, unsigned loc)
{ 
	char * dst = &data()[loc];
	memcpy( dst, src, size ); 
}

TypedBuffer::TypedBuffer() {}
TypedBuffer::~TypedBuffer() {}
	
void TypedBuffer::create(ValueType t, unsigned size)
{
	BaseBuffer::create(size);
	m_type = t;
}

TypedBuffer::ValueType TypedBuffer::valueType() const
{ return m_type; }
//:~
