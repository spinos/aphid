/*
 *  GAttrib.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Attrib.h"

namespace gar {

Attrib::Attrib(AttribName anm, AttribType at) :
m_anm(anm),
m_atyp(at)
{}

Attrib::~Attrib()
{}

const AttribName& Attrib::attrName() const
{ return m_anm; }

const AttribType& Attrib::attrType() const
{ return m_atyp; }

void Attrib::setValue(const bool& x)
{ memcpy(m_data, &x, sizeof(bool) ); }

void Attrib::setValue(const int& x)
{ memcpy(m_data, &x, 4 ); }

void Attrib::setValue(const float& x)
{ memcpy(m_data, &x, 4 ); }	

void Attrib::setValue2(const int* x)
{ memcpy(m_data, &x, 8 ); }

void Attrib::setValue2(const float* x)
{ memcpy(m_data, &x, 8 ); }

void Attrib::setValue3(const int* x)
{ memcpy(m_data, &x, 12 ); }

void Attrib::setValue3(const float* x)
{ memcpy(m_data, &x, 12 ); }

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
{ memcpy(&y, m_data, 8 ); }

void Attrib::getValue3(int* y) const
{ memcpy(&y, m_data, 12 ); }

void Attrib::getValue3(float* y) const
{ memcpy(&y, m_data, 12 ); }

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

static const char* sAttribNameAsStr[] = {
"unknown",
"grow portion",
"grow margin",
"grow angle",
"zenith noise",
"file name",
"width",
"height",
"radius",
"left side",
"right side",
"bend",
"twist",
"roll",
"width variation",
"height variation",
"radius variation",
};

const char* Attrib::attrNameStr() const
{ return sAttribNameAsStr[m_anm]; }

gar::AttribName Attrib::IntAsAttribName(int x)
{
	gar::AttribName r = gar::nUnknown;
	switch (x) {
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
		case gar::nWidthVariation :
			r = gar::nWidthVariation;
		break;
		case gar::nHeightVariation :
			r = gar::nHeightVariation;
		break;
		case gar::nRadiusVariation :
			r = gar::nRadiusVariation;
		break;
		default:
			;
	}
	return r;
}


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

