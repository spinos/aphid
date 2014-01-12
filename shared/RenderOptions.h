/*
 *  RenderOptions.h
 *  aphid
 *
 *  Created by jian zhang on 1/13/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class RenderOptions {
public:
	RenderOptions();
	virtual ~RenderOptions();
	
	void setAASample(int x);
	void setRenderImageWidth(int x);
	void setRenderImageHeight(int y);
	int AASample() const;
	int renderImageWidth() const;
	int renderImageHeight() const;
protected:

private:
	int m_resX, m_resY, m_AASample;
};