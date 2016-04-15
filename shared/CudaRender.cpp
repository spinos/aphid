/*
 *  CudaRender.cpp
 *  
 *
 *  Created by jian zhang on 3/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "CudaRender.h"
#include "CudaBase.h"
#include <iostream>
#include <cmath>
namespace aphid {

CudaRender::CudaRender(int tileSize) 
{
	CudaBase::SetDevice();
	
	const Vector3F eye(0.f, 0.f, 100.f);
	setEyePosition((float *)&eye);
	
	Matrix44F m;
	m.setTranslation(eye);
	*cameraSpaceR() = m;
	m.inverse();
	*cameraInvSpaceR() = m;

/// 35mm Academy
	std::cout<<"\n angle of view "<<180.f/3.14f*2.f * atan(21.9456f/2.f/35.f)<<" deg";
	setFrustum(.864f, .63f, 35.f, -1.f, -20000.f);
	m_tileSize = tileSize;
}

CudaRender::~CudaRender() {}

void CudaRender::setBufferSize(const int & w, const int & h)
{
	m_tileDim[0] = w / m_tileSize;
	m_tileDim[1] = h / m_tileSize;
	m_bufferLength = w * h;
	
/// uint in rgba
	m_hostColor.create( m_bufferLength * 4 );
	m_deviceColor.create( m_bufferLength * 4 );
/// float for near and far
	m_deviceDepth.create( m_bufferLength * 8 );
}

void CudaRender::render()
{}

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
{ return &hostColor()[(y * tileX() + x) * m_tileSize * m_tileSize]; }

/// copy in scanline
void CudaRender::sendTileColor(unsigned * dst, const int & stride,
						const int & x, const int & y) const
{
	unsigned * src = tileHostColor(x, y);
	
	for(int i=0; i<m_tileSize; ++i)
		memcpy ( &dst[i*stride], &src[i*m_tileSize], m_tileSize * 4 );
}

void CudaRender::sendImageColor(unsigned * dst, int len) const
{ memcpy ( dst, hostColor(), len * 4 ); }

void * CudaRender::colorBuffer()
{ return m_deviceColor.bufferOnDevice(); }

void CudaRender::getRoundedSize(int & w, int & h) const
{
	if(w<m_tileSize) w = m_tileSize;
	if(h<m_tileSize) h = m_tileSize;
	
    int tw = w / m_tileSize;
    if((w & (m_tileSize - 1)) > 0) tw++;
    w = tw * m_tileSize;
    
    int th = h / m_tileSize;
    if((h & (m_tileSize - 1)) > 0) th++;
    h = th * m_tileSize;
}

void CudaRender::colorToHost()
{ m_deviceColor.deviceToHost(m_hostColor.data(), bufferLength() * 4); }

void CudaRender::colorToHost(unsigned * dst, int npix)
{ m_deviceColor.deviceToHost(dst, npix * 4); }

const int & CudaRender::bufferLength() const
{ return m_bufferLength; }

void * CudaRender::nearDepthBuffer()
{ return m_deviceDepth.bufferOnDevice(); }

void * CudaRender::farDepthBuffer()
{ return m_deviceDepth.bufferOnDeviceAt( m_bufferLength * 4 ); }

const int & CudaRender::tileSize() const
{ return m_tileSize; }

}