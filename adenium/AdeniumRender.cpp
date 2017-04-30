#include "AdeniumRender.h"
#include <CUDABuffer.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <CudaBase.h>
#include <CudaPixelBuffer.h>
#include <iostream>
#include "AdeniumRenderInterface.h"
#include <PerspectiveCamera.h>
#include <BvhTriangleSystem.h>
AdeniumRender::AdeniumRender() :
m_imageWidth(0), m_imageHeight(0)
{
    m_deviceRgbaPix = new CudaPixelBuffer;
	m_depth = new CUDABuffer;
}

AdeniumRender::~AdeniumRender() 
{
    delete m_deviceRgbaPix;
	delete m_depth;
}

bool AdeniumRender::resize(int w, int h)
{ 
    if(!isSizeValid(w, h)) return false;
	if(w==m_imageWidth && h == m_imageHeight) return false;
    m_imageWidth = w;
    m_imageHeight = h;
    std::cout<<" resize render area: "<<w<<" x "<<h<<"\n";
	return true;
}

bool AdeniumRender::isSizeValid(int x, int y) const
{ return (x > 0 && y > 0); }

int AdeniumRender::numPixels() const
{ return m_imageWidth * m_imageHeight; }

void AdeniumRender::reset()
{
	m_deviceRgbaPix->create(numPixels() * 4);
	m_depth->create(numPixels() * 4);
	int imgs[2];
	imgs[0] = imageWidth();
	imgs[1] = imageHeight();
	adetrace::setImageSize(imgs);
	void * pix = m_deviceRgbaPix->map();
	void * d = m_depth->bufferOnDevice();
	adetrace::resetImage((uint *)pix, (float *)d, (uint)numPixels());
	CudaBase::CheckCudaError(" reset image");
	m_deviceRgbaPix->unmap();
}

void AdeniumRender::setModelViewMatrix(float * src)
{
	adetrace::setModelViewMatrix(src, 64);
}

void AdeniumRender::renderOrhographic(BaseCamera * camera, BvhTriangleSystem * tri)
{
	float camp[2];
	camp[0] = camera->fieldOfView();
	camp[1] = camera->aspectRatio();
	adetrace::setCameraProp(camp);
	
	void * internalNodeChildIndex = tri->internalNodeChildIndices();
	void * internalNodeAabbs = tri->internalNodeAabbs();
	void * indirection = tri->primitiveHash();
    void * pix = m_deviceRgbaPix->map();
    void * vertices = tri->deviceTretradhedronIndices(); 
    void * points = tri->deviceX();
	adetrace::renderImage((uint *)pix,
				(float *)m_depth->bufferOnDevice(),
                imageWidth(),
                imageHeight(),
                (int2 *)internalNodeChildIndex,
				(Aabb *)internalNodeAabbs,
				(KeyValuePair *)indirection,
				(int4 *)vertices,
				(float3 *)points,
				1);
	CudaBase::CheckCudaError(" render ortho image");
	m_deviceRgbaPix->unmap();
}

void AdeniumRender::renderPerspective(BaseCamera * camera, BvhTriangleSystem * tri)
{
	float camp[2];
	camp[0] = camera->frameWidth();
	camp[1] = camera->aspectRatio();
	adetrace::setCameraProp(camp);
	
	void * internalNodeChildIndex = tri->internalNodeChildIndices();
	void * internalNodeAabbs = tri->internalNodeAabbs();
	void * indirection = tri->primitiveHash();
    void * pix = m_deviceRgbaPix->map();
    void * vertices = tri->deviceTretradhedronIndices(); 
    void * points = tri->deviceX();
	adetrace::renderImage((uint *)pix,
				(float *)m_depth->bufferOnDevice(),
                imageWidth(),
                imageHeight(),
                (int2 *)internalNodeChildIndex,
				(Aabb *)internalNodeAabbs,
				(KeyValuePair *)indirection,
				(int4 *)vertices,
				(float3 *)points,
				0);
	CudaBase::CheckCudaError(" render persp image");
	m_deviceRgbaPix->unmap();
}

const int AdeniumRender::imageWidth() const
{ return m_imageWidth; }

const int AdeniumRender::imageHeight() const
{ return m_imageHeight; }

void AdeniumRender::bindBuffer()
{ m_deviceRgbaPix->bind(); }

void AdeniumRender::unbindBuffer()
{ m_deviceRgbaPix->unbind(); }
//:~