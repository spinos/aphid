/*
 *  RenderInterface.h
 *  
 *  access to camera, image, buffer, renderer, context
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef RENDER_INTERFACE_H
#define RENDER_INTERFACE_H

#include <QImage>

namespace aphid {
class BaseCamera;
}

class DisplayCamera;
class DisplayImage;
class DeepBuffer;
class BufferBlock;
class Renderer;
class RenderContext;

class RenderInterface {

	DisplayCamera* m_camera;
	DeepBuffer* m_buffer;
	DisplayImage* m_image;
	RenderContext* m_context;
	Renderer* m_renderer;
	int m_resizedImageDim[2];
	
public:
	RenderInterface();
	
	void setCamera(aphid::BaseCamera* x);
	void setChangedCamera();
	bool cameraChanged() const;
	
	bool imageSizeChanged() const;
	void createImage(int w, int h);
	void setResizedImage(int w, int h);
	int resizedImageWidth() const;
	int resizedImageHeight() const;
/// set frame in each block	
	void updateDisplayView();
/// by high residual
	BufferBlock* selectBlock();
	DisplayImage* image();
	
	uchar* imageScanline(int i);
	
	const int& xres() const;
	const int& yres() const;
	int bufferNumBlocks() const;
	
	Renderer* getRenderer();
	QImage getQImage() const;
	
	RenderContext* getContext();
	
/// quality threshold
	bool isResidualLowEnough() const;
	
};

#endif
