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
	const Vector3F eye(0.f, 0.f, 30.f);
	setEyePosition((float *)&eye);
	
	Matrix44F m;
	m.setTranslation(eye);
	*cameraSpaceP() = m;
	m.inverse();
	*cameraInvSpaceP() = m;
	
	setFrustum(1.33f, 1.f, 5.2f, -1.f, -1000.f);
}

CudaRender::~CudaRender() {}

void CudaRender::setBufferSize(const int & w, const int & h)
{
	m_tileDim[0] = w>>4;
	m_tileDim[1] = h>>4;
	m_bufferLength = w * h;
	
/// uint in rgba
	m_hostColor.create( m_bufferLength * 4 );
	m_deviceColor.create( m_bufferLength * 4 );
/// float
	m_deviceDepth.create( m_bufferLength * 4 );
}

void CudaRender::setPortSize(const int & w, const int & h)
{ setRect(w, h); }

const int & CudaRender::tileX() const
{ return m_tileDim[0]; }

const int & CudaRender::tileY() const
{ return m_tileDim[1]; }

int * CudaRender::tileDim()
{ return m_tileDim; }

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
{ m_deviceColor.deviceToHost(m_hostColor.data(), bufferLength() * 4); }

const int & CudaRender::bufferLength() const
{ return m_bufferLength; }

}