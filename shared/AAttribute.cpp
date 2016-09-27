/*
 *  AAttribute.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AAttribute.h"
#include <algorithm>
#include <memory.h>

namespace aphid {

AAttribute::AAttribute() 
{
	m_longName = "unknown";
	m_shortName = "unknown";
}

AAttribute::~AAttribute() {}

const TypedEntity::Type AAttribute::type() const
{ return TAttribute; }

AAttribute::AttributeType AAttribute::attrType() const
{ return aUnknown; }

std::string AAttribute::longName() const
{ return m_longName; }

std::string AAttribute::shortName() const
{ return m_shortName; }
	
void AAttribute::setLongName(const std::string & s)
{ m_longName = s; }

void AAttribute::setShortName(const std::string & s)
{ m_shortName = s; }

bool AAttribute::isNumeric() const
{ return attrType() == aNumeric; }

bool AAttribute::isEnum() const
{ return attrType() == aEnum; }

bool AAttribute::isString() const
{ return attrType() == aString; }

bool AAttribute::isCompound() const
{ return attrType() == aCompound; }

std::string AAttribute::attrTypeStr() const
{
	std::string r("unknown");
	switch (attrType()) {
		case aNumeric:
			r = "numeric";
			break;
		case aEnum:
			r = "enum";
			break;
		case aString:
			r = "string";
			break;
		case aCompound:
			r = "compound";
			break;
		default:
			break;
	}
	return r;
}

AStringAttribute::AStringAttribute() {}
AStringAttribute::~AStringAttribute() {}

AAttribute::AttributeType AStringAttribute::attrType() const
{ return aString; }

std::string AStringAttribute::value() const
{ return m_value; }

void AStringAttribute::setValue(const std::string & s)
{ m_value = s; }

AEnumAttribute::AEnumAttribute() {}
AEnumAttribute::~AEnumAttribute() {}

AAttribute::AttributeType AEnumAttribute::attrType() const
{ return aEnum; }

void AEnumAttribute::setRange(short a, short b)
{ m_minInd = a;
	m_maxInd = b;
}

void AEnumAttribute::setValue(short a)
{ m_value = a; }

void AEnumAttribute::addField(short ind, const std::string & name)
{ m_fields[ind] = name; }

short AEnumAttribute::value(short & a, short & b) const
{
	a = m_minInd;
	b = m_maxInd;
	return m_value;
}
std::string AEnumAttribute::fieldName(short ind)
{ return m_fields[ind]; }

unsigned AEnumAttribute::numFields() const
{ return m_fields.size(); }

short AEnumAttribute::asShort() const
{ return m_value; }

ACompoundAttribute::ACompoundAttribute() {}
ACompoundAttribute::~ACompoundAttribute() 
{
	std::vector<AAttribute *>::iterator it = m_children.begin();
	for(; it!= m_children.end(); ++it) delete *it;
	m_children.clear();
}

AAttribute::AttributeType ACompoundAttribute::attrType() const
{ return aCompound; }

void ACompoundAttribute::addChild(AAttribute * c)
{ m_children.push_back(c); }

unsigned ACompoundAttribute::numChild() const
{ return m_children.size(); }

AAttribute * ACompoundAttribute::child(unsigned idx)
{ return m_children[idx]; }

ANumericAttribute::ANumericAttribute() {}

ANumericAttribute::~ANumericAttribute() {}

AAttribute::AttributeType ANumericAttribute::attrType() const
{ return aNumeric; }

ANumericAttribute::NumericAttributeType ANumericAttribute::numericType() const
{ return TUnkownNumeric; }

void ANumericAttribute::setValue(const int & x) {}
void ANumericAttribute::setValue(const float & x) {}

AShortNumericAttribute::AShortNumericAttribute()
{ m_value = 0; }

AShortNumericAttribute::AShortNumericAttribute(short x)
{ m_value = x; }

AShortNumericAttribute::~AShortNumericAttribute() {}

ANumericAttribute::NumericAttributeType AShortNumericAttribute::numericType() const
{ return TShortNumeric; }

void AShortNumericAttribute::setValue(const int & x) 
{ m_value = x; }

void AShortNumericAttribute::setValue(const float & x) 
{ m_value = x; }

short AShortNumericAttribute::value() const
{ return m_value; }

AByteNumericAttribute::AByteNumericAttribute() {}

AByteNumericAttribute::AByteNumericAttribute(short x) : AShortNumericAttribute(x) {}

AByteNumericAttribute::~AByteNumericAttribute() {}

ANumericAttribute::NumericAttributeType AByteNumericAttribute::numericType() const
{ return TByteNumeric; }

char AByteNumericAttribute::asChar() const
{ return (char)value(); }

AIntNumericAttribute::AIntNumericAttribute()
{ m_value = 0; }

AIntNumericAttribute::AIntNumericAttribute(int x)
{ m_value = x; }

AIntNumericAttribute::~AIntNumericAttribute() {}

ANumericAttribute::NumericAttributeType AIntNumericAttribute::numericType() const
{ return TIntNumeric; }

void AIntNumericAttribute::setValue(const int & x) 
{ m_value = x; }

void AIntNumericAttribute::setValue(const float & x) 
{ m_value = x; }

int AIntNumericAttribute::value() const
{ return m_value; }

AFloatNumericAttribute::AFloatNumericAttribute() 
{ m_value = 0.f; }

AFloatNumericAttribute::AFloatNumericAttribute(float x) 
{ m_value = x; }

AFloatNumericAttribute::~AFloatNumericAttribute() {}

ANumericAttribute::NumericAttributeType AFloatNumericAttribute::numericType() const
{ return TFloatNumeric; }

void AFloatNumericAttribute::setValue(const int & x) 
{ m_value = x; }

void AFloatNumericAttribute::setValue(const float & x) 
{ m_value = x; }

float AFloatNumericAttribute::value() const
{ return m_value; }

ADoubleNumericAttribute::ADoubleNumericAttribute()
{ m_value = 0.; }

ADoubleNumericAttribute::ADoubleNumericAttribute(double x)
{ m_value = x; }

ADoubleNumericAttribute::~ADoubleNumericAttribute() {}

ANumericAttribute::NumericAttributeType ADoubleNumericAttribute::numericType() const
{ return TDoubleNumeric; }

void ADoubleNumericAttribute::setValue(const int & x) 
{ m_value = x; }

void ADoubleNumericAttribute::setValue(const float & x) 
{ m_value = x; }

void ADoubleNumericAttribute::setValue(const double & x)
{ m_value = x; }

double ADoubleNumericAttribute::value() const
{ return m_value; }

ABooleanNumericAttribute::ABooleanNumericAttribute() 
{ m_value = false; }

ABooleanNumericAttribute::ABooleanNumericAttribute(bool x) 
{ m_value = x; }

ABooleanNumericAttribute::~ABooleanNumericAttribute() {}

ANumericAttribute::NumericAttributeType ABooleanNumericAttribute::numericType() const
{ return TBooleanNumeric; }

void ABooleanNumericAttribute::setValue(const int & x) 
{ m_value = (x > 0); }

void ABooleanNumericAttribute::setValue(const float & x) 
{ m_value = (x > 0.f); }

bool ABooleanNumericAttribute::value() const
{ return m_value; }

char ABooleanNumericAttribute::asChar() const
{ return (char)value(); }

ABundleAttribute::ABundleAttribute() :
m_bundleSize(0),
m_stride(0),
m_ntyp(TUnkownNumeric),
m_v(NULL)
{}

ABundleAttribute::~ABundleAttribute()
{ if(m_v) delete[] m_v; }

AAttribute::AttributeType ABundleAttribute::attrType() const
{ return aNumericBundle; }

ANumericAttribute::NumericAttributeType ABundleAttribute::numericType() const
{ return m_ntyp; }

void ABundleAttribute::create(const int & sz,
	            NumericAttributeType ntyp)
{
    m_ntyp = ntyp;
    if(ntyp == TByteNumeric)
        m_stride = sizeof(char);
    else if(ntyp == TShortNumeric)
        m_stride = sizeof(short);
    else if(ntyp == TIntNumeric)
        m_stride = sizeof(int);
    else if(ntyp == TFloatNumeric)
        m_stride = sizeof(float);
    else if(ntyp == TDoubleNumeric)
        m_stride = sizeof(double);
    else if(ntyp == TBooleanNumeric)
        m_stride = sizeof(bool);
        
    m_v = new char[sz*m_stride];
    m_bundleSize = sz;
}

const char * ABundleAttribute::value() const
{ return m_v; }
	
char * ABundleAttribute::value()
{ return m_v; }

const int & ABundleAttribute::size() const
{ return m_bundleSize; }

const int ABundleAttribute::dataLength() const
{ return m_bundleSize * m_stride; }

void ABundleAttribute::setAttribValue(ANumericAttribute * a, const int & i)
{
    switch (a->numericType() ) {
        case TByteNumeric:
            setValue<char, AByteNumericAttribute>(
                static_cast<AByteNumericAttribute *>(a), i);
            break;
        case TShortNumeric:
            setValue<short, AShortNumericAttribute>(
                static_cast<AShortNumericAttribute *>(a), i);
            break;
        case TIntNumeric:
            setValue<int, AIntNumericAttribute>(
                static_cast<AIntNumericAttribute *>(a), i);
            break;
        case TFloatNumeric:
            setValue<float, AFloatNumericAttribute>(
                static_cast<AFloatNumericAttribute *>(a), i);
            break;
        case TDoubleNumeric:
            setValue<double, ADoubleNumericAttribute>(
                static_cast<ADoubleNumericAttribute *>(a), i);
            break;
        case TBooleanNumeric:
            setValue<bool, ABooleanNumericAttribute>(
                static_cast<ABooleanNumericAttribute *>(a), i);
            break;
        default:
            break;
    }
}

void ABundleAttribute::getAttribValue(ANumericAttribute * dst, 
                                        const int & i) const
{
    switch (numericType() ) {
        case TByteNumeric:
            getValue<char, AByteNumericAttribute>(
                static_cast<AByteNumericAttribute *>(dst), i);
            break;
        case TShortNumeric:
            getValue<short, AShortNumericAttribute>(
                static_cast<AShortNumericAttribute *>(dst), i);
            break;
        case TIntNumeric:
            getValue<int, AIntNumericAttribute>(
                static_cast<AIntNumericAttribute *>(dst), i);
            break;
        case TFloatNumeric:
            getValue<float, AFloatNumericAttribute>(
                static_cast<AFloatNumericAttribute *>(dst), i);
            break;
        case TDoubleNumeric:
            getValue<double, ADoubleNumericAttribute>(
                static_cast<ADoubleNumericAttribute *>(dst), i);
            break;
        case TBooleanNumeric:
            getValue<bool, ABooleanNumericAttribute>(
                static_cast<ABooleanNumericAttribute *>(dst), i);
            break;
        default:
            break;
    }
}

void ABundleAttribute::getNumericAttrib(ANumericAttribute * & dst) const
{
    switch (numericType() ) {
        case TByteNumeric:
            dst = new AByteNumericAttribute;
            break;
        case TShortNumeric:
            dst = new AShortNumericAttribute;
            break;
        case TIntNumeric:
            dst = new AIntNumericAttribute;
            break;
        case TFloatNumeric:
            dst = new AFloatNumericAttribute;
            break;
        case TDoubleNumeric:
            dst = new ADoubleNumericAttribute;
            break;
        case TBooleanNumeric:
            dst = new ABooleanNumericAttribute;
            break;
        default:
            break;
    }
    
    if(!dst)
        return;
        
    dst->setShortName(shortName() );
    dst->setLongName(longName() );
}
	
}
//:~
