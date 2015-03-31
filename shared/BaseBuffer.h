#ifndef BASEBUFFER_H
#define BASEBUFFER_H

/*
 *  BaseBuffer.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

class BaseBuffer {
public:
	enum BufferType {
		kUnknown = 0,
	    kOnHost = 1,
		kVBO = 2,
		kOnDevice = 3
	};
	
	BaseBuffer();
	virtual ~BaseBuffer();
	
	virtual void create(float * data, unsigned size);
	virtual void create(unsigned size);
	virtual void destroy();
	
	const unsigned bufferName() const;
	const unsigned bufferSize() const;
	char * data() const;

protected:
	void createVBO(float * data, unsigned size);
	void destroyVBO();
	
	void setBufferType(BufferType t);
	const BufferType bufferType() const;
	void setBufferSize(unsigned size);
	bool canResize(unsigned n);
	
private:
    char * m_native;
	unsigned m_bufferName;
	unsigned m_bufferSize;
	BufferType m_bufferType;
};
#endif        //  #ifndef BASEBUFFER_H
