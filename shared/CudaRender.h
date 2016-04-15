/*
 *  CudaRender.h
 *  
 *
 *  Created by jian zhang on 3/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef CUDARENDER_H
#define CUDARENDER_H

#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <ViewCull.h>

namespace aphid {

class CudaRender : public BaseView {

	BaseBuffer m_hostColor;
	CUDABuffer m_deviceColor;
	CUDABuffer m_deviceDepth;
	int m_tileDim[2];
	int m_tileSize;
	int m_bufferLength;
	
public :
	CudaRender(int m_tileSize = 16);
	virtual ~CudaRender();
	
	virtual void setBufferSize(const int & w, const int & h);
	virtual void render();
	
	void setPortSize(const int & w, const int & h);
	const int & tileX() const;
	const int & tileY() const;
	unsigned * hostColor() const;
	unsigned * tileHostColor(const int & x, const int & y) const;
	void sendTileColor(unsigned * dst, const int & stride,
					const int & x, const int & y) const;
	void sendImageColor(unsigned * dst, int len) const;
	
	void getRoundedSize(int & w, int & h) const;
	const int & tileSize() const;
	
	void colorToHost();
	
protected :
	void * nearDepthBuffer();
	void * farDepthBuffer();
	void * colorBuffer();
	int * tileDim();
	const int & bufferLength() const;
	
private:

};

}

#endif