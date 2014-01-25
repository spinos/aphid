/*
 *  TextureGroup.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "TextureGroup.h"
#include "BaseTexture.h"

TextureGroup::TextureGroup() 
{
	m_activeTex = 0;
}

TextureGroup::~TextureGroup()
{
	clearTextures();
}

void TextureGroup::addTexture(BaseTexture * l)
{
	m_texs.push_back(l);
}

unsigned TextureGroup::numTextures() const
{
	return m_texs.size();
}

BaseTexture * TextureGroup::getTexture(unsigned idx) const
{
	return m_texs[idx];
}

BaseTexture * TextureGroup::getTexture(const std::string & name) const
{
	std::vector<BaseTexture *>::const_iterator it = m_texs.begin();
	for(; it != m_texs.end(); ++it) {
		if((*it)->name() == name)
			return *it;
	}
	return NULL;
}

void TextureGroup::clearTextures()
{
	std::vector<BaseTexture *>::iterator it = m_texs.begin();
	for(; it != m_texs.end(); ++it) {
		delete *it;
	}
	m_texs.clear();
}

char TextureGroup::selectTexture(unsigned idx)
{
	if(idx >= numTextures()) return 0;
	m_activeTex = m_texs[idx];
	return 1;
}

BaseTexture * TextureGroup::selectedTexture() const
{
	return m_activeTex;
}