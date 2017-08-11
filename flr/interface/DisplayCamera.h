/*
 *  DisplayCamera.h
 *  
 *  camera view frame projection
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DISPLAY_CAMERA_H
#define DISPLAY_CAMERA_H

namespace aphid {
class Matrix44F;
class BaseCamera;
}

class BufferBlock;

class DisplayCamera {

	aphid::Matrix44F* m_lastViewFrame;
	aphid::BaseCamera* m_camera;
/// left-top origin is (-m_frameCenterX, -m_frameCenterY)
	int m_frameCenterX, m_frameCenterY;
/// per-pixel
	float m_deltaX, m_deltaY;
	
public:
	DisplayCamera();
	~DisplayCamera();
	
	void setCamera(aphid::BaseCamera* x);
	void setFrameCenter(int x, int y);
	
	bool isViewFrameChanged() const;
	void updateViewFrame();
/// calculte frame of block
	void setBlockView(BufferBlock* blk);
	
};

#endif
