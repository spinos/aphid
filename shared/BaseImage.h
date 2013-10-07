/*
 *  BaseImage.h
 *  arum
 *
 *  Created by jian zhang on 9/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BaseFile.h>
class BaseImage : public BaseFile {
public:
	enum ChannelRank {
		RGB = 3,
		RGBA = 4
	};
	
	BaseImage();
	virtual ~BaseImage();
	
	virtual char load(const char * filename);
	virtual void clear();
	virtual const char * formatName() const;
	virtual void allWhite();
	virtual void allBlack();
	
	char isValid() const;
	int getWidth() const;
	int getHeight() const;
	int pixelLoc(float s, float t, bool flipV = false) const;

	virtual float sampleRed(float u, float v);
	virtual float sampleRed(float u, float v) const;
	virtual void setRed(float u, float v, float red);
	virtual void applyMask(BaseImage * another);
	void verbose() const;
	
	int m_imageWidth, m_imageHeight;
	ChannelRank m_channelRank;
	char _valid;
};