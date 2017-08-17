/*
 *  GAttrib.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Attrib.h"
#include <iostream>

namespace gar {

Attrib::Attrib(AttribName anm, AttribType atyp)
{ 
	m_anm = anm;
	m_atyp = atyp;
}

Attrib::~Attrib()
{}

AttribName Attrib::attrName() const
{ return m_anm; }

AttribType Attrib::attrType() const
{ return m_atyp; }

void Attrib::setValue(const bool& x)
{ memcpy(m_data, &x, sizeof(bool) ); }

void Attrib::setValue(const int& x)
{ memcpy(m_data, &x, 4 ); }

void Attrib::setValue(const float& x)
{ memcpy(m_data, &x, 4 ); }	

void Attrib::setValue2(const int* x)
{ memcpy(m_data, x, 8 ); }

void Attrib::setValue2(const float* x)
{ memcpy(m_data, x, 8 ); }

void Attrib::setValue3(const int* x)
{ memcpy(m_data, x, 12 ); }

void Attrib::setValue3(const float* x)
{ memcpy(m_data, x, 12 ); }

void Attrib::setMin(const int& x)
{ memcpy(&m_data[8], &x, 4 ); }

void Attrib::setMin(const float& x)
{ memcpy(&m_data[8], &x, 4 ); }

void Attrib::setMax(const int& x)
{ memcpy(&m_data[16], &x, 4 ); }

void Attrib::setMax(const float& x)
{ memcpy(&m_data[16], &x, 4 ); }

void Attrib::getValue(bool& y) const
{ memcpy(&y, m_data, sizeof(bool) ); }

void Attrib::getValue(int& y) const
{ memcpy(&y, m_data, 4 ); }

void Attrib::getValue(float& y) const
{ memcpy(&y, m_data, 4 ); }

void Attrib::getValue2(int* y) const
{ memcpy(&y, m_data, 8 ); }

void Attrib::getValue2(float* y) const
{ memcpy(y, m_data, 8 ); }

void Attrib::getValue3(int* y) const
{ memcpy(y, m_data, 12 ); }

void Attrib::getValue3(float* y) const
{ memcpy(y, m_data, 12 ); }

void Attrib::getMin(int& y) const
{ memcpy(&y, &m_data[8], 4 ); }

void Attrib::getMin(float& y) const
{ memcpy(&y, &m_data[8], 4 ); }

void Attrib::getMax(int& y) const
{ memcpy(&y, &m_data[16], 4 ); }

void Attrib::getMax(float& y) const
{ memcpy(&y, &m_data[16], 4 ); }

bool Attrib::isStringType() const
{ return m_atyp == tString; }

const char* Attrib::sAttribNameAsStr[] = {
"unknown",
"node name",
"grow portion",
"grow margin",
"grow angle",
"zenith noise",
"file name",
"width",
"height",
"radius",
"center line",
"left side",
"right side",
"bend",
"twist",
"roll",
"fold",
"crumple",
"width variation",
"height variation",
"radius variation",
"size variation",
"bend variation",
"weight variation",
"noise variation",
"crumple variation",
"fold variation",
"aging variation",
"petiole angle",
"leaf placement",
"whorl count",
};

std::string Attrib::attrNameStr() const
{ 
	if(m_anm > 2048 + 30) {
		std::cout<<"\n ERROR oor attr name "<<m_anm;
		std::cout.flush();
		return std::string("unknown");
	}
	return std::string(sAttribNameAsStr[m_anm - 2048]); 
}

gar::AttribName Attrib::IntAsAttribName(int x)
{
	gar::AttribName r = gar::nUnknown;
	switch (x) {
		case gar::nmNodeName :
			r = gar::nmNodeName;
		break;
	    case gar::nGrowPortion :
			r = gar::nGrowPortion;
		break;
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
		case gar::nWidth :
			r = gar::nWidth;
		break;
		case gar::nHeight :
			r = gar::nHeight;
		break;
		case gar::nRadius :
			r = gar::nRadius;
		break;
		case gar::nCenterLine :
			r = gar::nCenterLine;
		break;
		case gar::nLeftSide :
			r = gar::nLeftSide;
		break;
		case gar::nRightSide :
			r = gar::nRightSide;
		break;
		case gar::nBend :
			r = gar::nBend;
		break;
		case gar::nTwist :
			r = gar::nTwist;
		break;
		case gar::nRoll :
			r = gar::nRoll;
		break;
		case gar::nFold :
			r = gar::nFold;
		break;
		case gar::nmCrumple :
			r = gar::nmCrumple;
		break;
		case gar::nWidthVariation :
			r = gar::nWidthVariation;
		break;
		case gar::nHeightVariation :
			r = gar::nHeightVariation;
		break;
		case gar::nRadiusVariation :
			r = gar::nRadiusVariation;
		break;
		case gar::nSizeVariation :
			r = gar::nSizeVariation;
		break;
		case gar::nBendVariation :
			r = gar::nBendVariation;
		break;
		case gar::nWeightVariation :
			r = gar::nWeightVariation;
		break;
		case gar::nNoiseVariation :
			r = gar::nNoiseVariation;
		break;
		case gar::nCrumpleVar :
			r = gar::nCrumpleVar;
		break;
		case gar::nFoldVar :
			r = gar::nFoldVar;
		break;
		case gar::nPetioleAngle :
			r = gar::nPetioleAngle;
		break;
		case gar::nLeafPlacement :
			r = gar::nLeafPlacement;
		break;
		case gar::nWhorlCount :
			r = gar::nWhorlCount;
		break;
		default:
			;
	}
	return r;
}

const char* Attrib::IntAsEnumFieldName(int x)
{
	switch (x) {
		case gar::phOpposite :
			return "opposite";
		break;
		case gar::phAlternate :
			return "alternate";
		break;
		case gar::phDecussate :
			return "decussate";
		break;
		case gar::phWhorled :
			return "whorled";
		break;
		default:
			;
	}
	return "unknown";
}

EnumAttrib::EnumAttrib(AttribName anm) : Attrib(anm, tEnum),
m_fields(NULL),
m_numFields(0)
{}

EnumAttrib::~EnumAttrib()
{
	if(m_fields)
		delete[] m_fields;
}

void EnumAttrib::createFields(int n)
{ 
	m_numFields = n;
	m_fields = new int[n]; 
}

void EnumAttrib::setField(int i, int x)
{ m_fields[i] = x; }

const int& EnumAttrib::numFields() const
{ return m_numFields; }

const int& EnumAttrib::getField(int x) const
{ return m_fields[x]; }


SplineAttrib::SplineAttrib(AttribName anm) : Attrib(anm, tSpline)
{
	m_splineValue[0] = 1.f;
	m_splineValue[1] = 1.f;
	m_splineCv[0] = .4f;
	m_splineCv[1] = 1.f;
	m_splineCv[2] = .6f;
	m_splineCv[3] = 1.f;
}

SplineAttrib::~SplineAttrib()
{}

void SplineAttrib::setSplineValue(float y0, float y1)
{
	m_splineValue[0] = y0;
	m_splineValue[1] = y1;
}

void SplineAttrib::setSplineCv0(float x, float y)
{
	m_splineCv[0] = x;
	m_splineCv[1] = y;
}

void SplineAttrib::setSplineCv1(float x, float y)
{
	m_splineCv[2] = x;
	m_splineCv[3] = y;
}

void SplineAttrib::getSplineValue(float* y) const
{ memcpy(y, m_splineValue, 8); }

void SplineAttrib::getSplineCv0(float* y) const
{ memcpy(y, m_splineCv, 8); }

void SplineAttrib::getSplineCv1(float* y) const
{ memcpy(y, &m_splineCv[2], 8); }

StringAttrib::StringAttrib(AttribName anm, const bool& ifn) : Attrib(anm, tString)
{ m_isFileName = ifn; }

StringAttrib::~StringAttrib()
{}

void StringAttrib::setValue(const std::string& x)
{ m_strdata = x; }

void StringAttrib::getValue(std::string& y) const
{ y = m_strdata; }

const bool& StringAttrib::isFileName() const
{ return m_isFileName; }

}

