/*
 *  BaseTexture.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/25/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseTexture.h"

BaseTexture::BaseTexture() { setAllWhite(true); }

BaseTexture::~BaseTexture() {}

unsigned BaseTexture::numTexels() const { return 0; }

void * BaseTexture::data() { return 0; }
void * BaseTexture::data() const { return 0; }

void BaseTexture::sample(const unsigned & /*faceIdx*/, float * /*dst*/) const {}

void BaseTexture::sample(const unsigned & /*faceIdx*/, const float & /*faceU*/, const float & /*faceV*/, float * /*dst*/) const {}

void BaseTexture::setTextureFormat(BaseTexture::TexelFormat f) { m_format = f; }

void BaseTexture::setTextureDepth(BaseTexture::TexelDepth d) { m_depth = d; }

BaseTexture::TexelFormat BaseTexture::textureFormat() const { return m_format; }

BaseTexture::TexelDepth BaseTexture::textureDepth() const { return m_depth; }

void BaseTexture::setAllWhite(bool x) { m_allWhite = x; }
bool BaseTexture::allWhite() const { return m_allWhite; }

unsigned BaseTexture::dataSize() const { return numTexels() * m_format * m_depth; }

const TypedEntity::Type BaseTexture::type() const
{ return TTexture; }