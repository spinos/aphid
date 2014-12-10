/*
 *  BaseBuffer.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class BaseBuffer {
public:
	enum BufferType {
		kUnknown = 0,
		kVBO = 1,
		kSimple = 2
	};
	
	BaseBuffer();
	virtual ~BaseBuffer();
	
	virtual void create(float * data, unsigned size);
	virtual void destroy();
	
	const unsigned bufferName() const;
	const unsigned bufferSize() const;

protected:
	void createVBO(float * data, unsigned size);
	void destroyVBO();
	
	void setBufferType(BufferType t);
	const BufferType bufferType() const;
	
private:
	BufferType m_bufferType;
    unsigned m_bufferName;
	unsigned m_bufferSize;
};