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
	Vector3F m_rayFrameVec[6];
	int m_tileDim[2];
	int m_imageDim[2];
	
public :
	CudaRender();
	virtual ~CudaRender();
	
	virtual void setSize(const int & w, const int & h);
	void setImageSize(const int & w, const int & h);
	const int & imageWidth() const;
	const int & imageHeight() const;
	const int & tileX() const;
	const int & tileY() const;
	unsigned * hostColor() const;
	unsigned * tileHostColor(const int & x, const int & y) const;
	void sendTileColor(unsigned * dst, const int & stride,
					const int & x, const int & y) const;
	
	static void GetRoundedSize(int & w, int & h);
	
protected :
	void * depthBuffer();
	void * colorBuffer();
	int * tileDim();
	int * imageDim();
	void colorToHost();
	void updateRayFrameVec();
	Vector3F * rayFrameVec();
	
private:

};

}

#endif