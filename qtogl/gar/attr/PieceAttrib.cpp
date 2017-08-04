/*
 *  PieceAttrib.cpp
 *  
 *
 *  Created by jian zhang on 8/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "PieceAttrib.h"
#include "gar_common.h"

PieceAttrib::PieceAttrib()
{
	char b[17];
    gar::GenGlyphName(b);
	m_glyphName = std::string(b);
}

PieceAttrib::~PieceAttrib()
{}

const std::string& PieceAttrib::glyphName() const
{ return m_glyphName; }
