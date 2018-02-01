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
/// 00 - 10
/// |    | 
/// 01 - 11
	float m_framePoints[4][3];
/// per-pixel
	float m_deltaX, m_deltaY;
	bool m_isChanged;
	
public:
	DisplayCamera();
	~DisplayCamera();
	
	void setCamera(aphid::BaseCamera* x);
	void setChanged();
	bool isChanged() const;
	
	void updateViewFrame();
/// calculte view frame of block
	void setBlockView(BufferBlock* blk) const;
	
private:
/// by pixel location
	void getPointOnFrame(float* pnt, const int& px, const int& py) const;
	
};

#endif
