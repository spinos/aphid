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
{
	AttribArrayTyp::iterator it = m_collAttrs.begin();
	for(;it!=m_collAttrs.end();++it) {
		delete *it;
	}
	m_collAttrs.clear();
}

const std::string& PieceAttrib::glyphName() const
{ return m_glyphName; }

void PieceAttrib::addIntAttrib(gar::AttribName anm,
		const int& val, 
		const int& minVal,
		const int& maxVal)
{
	gar::Attrib* aat = new gar::Attrib(anm, gar::tInt);
	aat->setValue(val);
	aat->setMin(minVal);
	aat->setMax(maxVal);
	m_collAttrs.push_back(aat);
	
}	

void PieceAttrib::addFloatAttrib(gar::AttribName anm,
		const float& val, 
		const float& minVal,
		const float& maxVal)
{
	gar::Attrib* aat = new gar::Attrib(anm, gar::tFloat);
	aat->setValue(val);
	aat->setMin(minVal);
	aat->setMax(maxVal);
	m_collAttrs.push_back(aat);
	
}

void PieceAttrib::addStringAttrib(gar::AttribName anm,
		const std::string& val,
		const bool& asFileName)
{
	gar::StringAttrib* aat = new gar::StringAttrib(anm, asFileName);
	aat->setValue(val);
	m_collAttrs.push_back(aat);
}

int PieceAttrib::numAttribs() const
{ return m_collAttrs.size(); }

gar::Attrib* PieceAttrib::getAttrib(const int& i)
{ return m_collAttrs[i]; }

const gar::Attrib* PieceAttrib::getAttrib(const int& i) const
{ return m_collAttrs[i]; }

gar::Attrib* PieceAttrib::findAttrib(gar::AttribName anm)
{
	AttribArrayTyp::iterator it = m_collAttrs.begin();
	for(;it!=m_collAttrs.end();++it) {
		if( (*it)->attrName() == anm)
			return *it;
	}
	return NULL;
}

gar::AttribName PieceAttrib::IntAsAttribName(int x)
{
	gar::AttribName r = gar::nUnknown;
	switch (x) {
		case gar::nGrowMargin :
			r = gar::nGrowMargin;
		break;
		case gar::nGrowAngle :
			r = gar::nGrowAngle;
		break;
		case gar::nZenithNoise :
			r = gar::nZenithNoise;
		break;
		case gar::nFileName :
			r = gar::nFileName;
		break;
		default:
			;
	}
	return r;
}
