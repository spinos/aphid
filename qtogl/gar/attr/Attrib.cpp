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
"grow margin",
"grow angle",
"zenith noise",
"file name"
};

const char* Attrib::attrNameStr() const
{ return sAttribNameAsStr[m_anm]; }

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

