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
	virtual void setMaxSubdiv(int x);
	virtual void setUseDisplaySize(bool x);
	int AASample() const;
	int renderImageWidth() const;
	int renderImageHeight() const;
	int maxSubdiv() const;
	bool useDisplaySize() const;
protected:

private:
	int m_resX, m_resY, m_AASample, m_maxSubdiv;
	bool m_useDisplaySize;
};