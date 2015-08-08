/*
 *  BaseBuffer.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "BaseBuffer.h"
#include <Vector3F.h>
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

void BaseBuffer::copyFrom(const void * src)
{ memcpy( m_native, src, bufferSize() ); }

void BaseBuffer::copyFrom(const void * src, unsigned size)
{ memcpy( m_native, src, size ); }

void BaseBuffer::copyFrom(const void * src, unsigned size, unsigned loc)
{ 
	char * dst = &((char *)m_native)[loc];
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

unsigned TypedBuffer::numElements() const
{
	unsigned n = bufferSize();
	switch (valueType()) {
		case TShort:
			n = n>>1;
			break;
		case TFlt:
			n = n>>2;
			break;
		case TVec2:
			n = n>>3;
			break;
		case TVec3:
			n = n/12;
			break;
		case TVec4:
			n = n>>4;
			break;
		default:
			break;
	}
	return n;
}

void TypedBuffer::operator-=( const BaseBuffer * other )
{
    switch (valueType()) {
		case TFlt:
            minusFlt(other);
			break;
		case TVec3:
            minusVec3(other);
			break;
		default:
			break;
	}
}

void TypedBuffer::minusFlt(const BaseBuffer * other)
{ minus<float>(other, numElements()); }

void TypedBuffer::minusVec3(const BaseBuffer * other)
{ minus<Vector3F>(other, numElements()); }
//:~