/*
 *  CudaRender.cpp
 *  
 *
 *  Created by jian zhang on 3/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "CudaRender.h"
#include <iostream>
namespace aphid {

CudaRender::CudaRender() 
{
	const Vector3F eye(0.f, 0.f, 50.f);
	setEyePosition((float *)&eye);
	
	Matrix44F m;
	m.setTranslation(eye);
	*cameraSpaceP() = m;
	m.inverse();
	*cameraInvSpaceP() = m;
	
	setFrustum(1.f, .75f, .732f, -1.f, -1000.f);
}

CudaRender::~CudaRender() {}

void CudaRender::setSize(const int & w, const int & h)
{
	setRect(w, h);
	m_tileDim[0] = w>>4;
	m_tileDim[1] = h>>4;
	int npix = w * h;
	
/// uint in rgba
	m_hostColor.create( npix * 4 );
	m_deviceColor.create( npix * 4 );
/// float
	m_deviceDepth.create( npix * 4 );
}

void CudaRender::setImageSize(const int & w, const int & h)
{ 
    m_imageDim[0] = w;
    m_imageDim[1] = h;
}

const int & CudaRender::imageWidth() const
{ return m_imageDim[0]; }

const int & CudaRender::imageHeight() const
{ return m_imageDim[1]; }

const int & CudaRender::tileX() const
{ return m_tileDim[0]; }

const int & CudaRender::tileY() const
{ return m_tileDim[1]; }

int * CudaRender::tileDim()
{ return m_tileDim; }

int * CudaRender::imageDim()
{ return m_imageDim; }

unsigned * CudaRender::hostColor() const
{ return (unsigned *)m_hostColor.data(); }

unsigned * CudaRender::tileHostColor(const int & x, const int & y) const
{ return &hostColor()[(y * tileX() + x)<<8]; }

/// copy in scanline
void CudaRender::sendTileColor(unsigned * dst, const int & stride,
						const int & x, const int & y) const
{
	unsigned * src = tileHostColor(x, y);
	
	for(int i=0; i<16; ++i)
		memcpy ( &dst[i*stride], &src[i*16], 64 );
}

void * CudaRender::depthBuffer()
{ return m_deviceDepth.bufferOnDevice(); }

void * CudaRender::colorBuffer()
{ return m_deviceColor.bufferOnDevice(); }

void CudaRender::GetRoundedSize(int & w, int & h)
{
	if(w<16) w = 16;
	if(h<16) h = 16;
/// round to 16
    int tw = w >> 4;
    if((w & 15) > 0) tw++;
    w = tw << 4;
    
    int th = h >> 4;
    if((h & 15) > 0) th++;
    h = th << 4;
}

void CudaRender::colorToHost()
{ m_deviceColor.deviceToHost(m_hostColor.data(), numPixels() * 4); }

void CudaRender::updateRayFrameVec()
{ frustum().toRayFrame(m_rayFrameVec); }

Vector3F * CudaRender::rayFrameVec()
{ return m_rayFrameVec; }

}