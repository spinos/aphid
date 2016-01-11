/*
 *  AAttribute.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AAttribute.h"

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
//:~