/*
 *  RenderEngine.h
 *  aphid
 *
 *  Created by jian zhang on 12/31/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
class BaseCamera;
class RenderEngine {
public:
	RenderEngine();
	virtual ~RenderEngine();
	
	void setCamera(BaseCamera * camera);
	void setResolution(unsigned resx, unsigned resy);
	
	unsigned resolutionX() const;
	unsigned resolutionY() const;
	
	virtual void render();
protected:

private:
	BaseCamera* m_camera;
	unsigned m_resolutionX, m_resolutionY;
};