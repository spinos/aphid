/*
 *  RenderInterface.cpp
 *  
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "RenderInterface.h"
#include "BufferBlock.h"
#include "DeepBuffer.h"
#include "DisplayImage.h"
#include "DisplayCamera.h"
#include "NoiseRenderer.h"
#include "BoxRenderer.h"
#include "RenderContext.h"
#include <math/BaseCamera.h>

using namespace aphid;

RenderInterface::RenderInterface()
{
	m_camera = new DisplayCamera;
	m_buffer = new DeepBuffer;
	m_image = new DisplayImage;
	m_context = new RenderContext;
	m_context->createSampler();
	m_context->createProjector();
#if 0
	m_renderer = new NoiseRenderer;
#else
	m_renderer = new BoxRenderer;
#endif 
}

void RenderInterface::setCamera(BaseCamera* x)
{ m_camera->setCamera(x); }

void RenderInterface::setChangedCamera()
{ m_camera->setChanged(); }

bool RenderInterface::cameraChanged() const
{ return m_camera->isChanged(); }

bool RenderInterface::imageSizeChanged() const
{
	return (xres() != m_resizedImageDim[0]
	|| yres() != m_resizedImageDim[1] );
}

void RenderInterface::createImage(int w, int h)
{ 
	m_buffer->create(w, h);
	
	int rxres = m_buffer->width();
	int ryres = m_buffer->height();
	m_image->create(rxres, ryres);
	m_resizedImageDim[0] = rxres;
	m_resizedImageDim[1] = ryres;
}

void RenderInterface::setResizedImage(int w, int h)
{
	m_resizedImageDim[0] = w;
	m_resizedImageDim[1] = h;
}

int RenderInterface::resizedImageWidth() const
{ return m_resizedImageDim[0]; }

int RenderInterface::resizedImageHeight() const
{ return m_resizedImageDim[1]; }

void RenderInterface::updateDisplayView()
{
	m_camera->updateViewFrame();
	m_buffer->setBegin(m_camera);
}

DisplayImage* RenderInterface::image()
{ return m_image; }

QImage RenderInterface::getQImage() const
{ return QImage(m_image->bits(), 
	m_image->xres(), m_image->yres(), 
	QImage::Format_RGB32); 
}

uchar* RenderInterface::imageScanline(int i)
{ return m_image->scanline(i); }

const int& RenderInterface::xres() const
{ return m_image->xres(); }
	
const int& RenderInterface::yres() const
{ return m_image->yres(); }

BufferBlock* RenderInterface::selectBlock()
{ return m_buffer->highResidualBlock(); }

int RenderInterface::bufferNumBlocks() const
{ return m_buffer->numBlocks(); }

Renderer* RenderInterface::getRenderer()
{ return m_renderer; }

RenderContext* RenderInterface::getContext()
{ return m_context; }

bool RenderInterface::isResidualLowEnough() const
{ return m_buffer->maxResidual() < .0008f; }
