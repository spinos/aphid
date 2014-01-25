/*
 *  TextureGroup.h
 *  aphid
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <vector>
class BaseTexture;
class TextureGroup {
public:
	TextureGroup();
	virtual ~TextureGroup();
	void addTexture(BaseTexture * l);
	unsigned numTextures() const;
	BaseTexture * getTexture(unsigned idx) const;
	BaseTexture * getTexture(const std::string & name) const;
	void clearTextures();
	char selectTexture(unsigned idx);
	BaseTexture * selectedTexture() const;
protected:

private:
	std::vector<BaseTexture *> m_texs;
	BaseTexture * m_activeTex;
};