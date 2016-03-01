#ifndef CUDABUFFER_H
#define CUDABUFFER_H

/*
 *  CUDABuffer.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
namespace aphid {

class CUDABuffer {
public:
	CUDABuffer();
	virtual ~CUDABuffer();
	
	void destroy();
	void create(unsigned size);
	
	void * bufferOnDevice();
	void * bufferOnDeviceAt(unsigned loc);
	
	void hostToDevice(void * src, unsigned size);
	void deviceToHost(void * dst, unsigned size);
	void hostToDevice(void * src);
	void deviceToHost(void * dst);
	void hostToDevice(void * src, unsigned loc, unsigned size);
	void deviceToHost(void * dst, unsigned loc, unsigned size);
	
	const unsigned bufferSize() const;
private:
    const unsigned minimalMemSize(unsigned size) const;
	
private:
	void *_device_vbo_buffer;
	unsigned m_bufferSize, m_reseveSize;
};

}
#endif        //  #ifndef CUDABUFFER_H

