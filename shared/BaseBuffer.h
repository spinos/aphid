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
	BaseBuffer();
	virtual ~BaseBuffer();
	
	void create(unsigned size);
	void destroy();
	
	const unsigned bufferName() const;
	const unsigned bufferSize() const;
	char * data() const;
    void copyFrom(const void * src, unsigned size);
	void copyFrom(const void * src, unsigned size, unsigned loc);
protected:

private:
    char * m_native;
	unsigned m_bufferSize;
};
#endif        //  #ifndef BASEBUFFER_H
