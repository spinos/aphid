/*
 *  AAttribute.h
 *  aphid
 *
 *  Created by jian zhang on 10/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "TypedEntity.h"
#include <string>
#include <vector>
#include <map>

class AAttribute : public TypedEntity {
public:
	enum AttributeType {
	    aUnknown,
	    aNumeric,
	    aEnum,
	    aTyped,
	    aCompound,
	    aString,
		aUnit
	};
	
	AAttribute();
	virtual ~AAttribute();
// override typed
	const TypedEntity::Type type() const;
	
	virtual AttributeType attrType() const;
	
	std::string longName() const;
	std::string shortName() const;
	
	void setLongName(const std::string & s);
	void setShortName(const std::string & s);
	
	bool isNumeric() const;
	bool isEnum() const;
	bool isString() const;
	bool isCompound() const;
	std::string attrTypeStr() const;
	
protected:
	
private:
	std::string m_longName, m_shortName;
};

class AStringAttribute : public AAttribute {
public:
	AStringAttribute();
	virtual ~AStringAttribute();
	
	virtual AttributeType attrType() const;
	
	std::string value() const;
	void setValue(const std::string & s);
protected:

private:
	std::string m_value;
};

class AEnumAttribute : public AAttribute {
public:
	AEnumAttribute();
	virtual ~AEnumAttribute();
	
	virtual AttributeType attrType() const;
	
	void setRange(short a, short b);
	void setValue(short a);
	void addField(short ind, const std::string & name);
	
	unsigned numFields() const;
	short value(short & a, short & b) const;
	std::string fieldName(short ind);
	short asShort() const;
	
protected:

private:
	short m_value, m_minInd, m_maxInd;
	std::map<short, std::string > m_fields;
};

class ACompoundAttribute : public AAttribute {
public:
	ACompoundAttribute();
	virtual ~ACompoundAttribute();
	
	virtual AttributeType attrType() const;
	
	void addChild(AAttribute * c);
	unsigned numChild() const;
	AAttribute * child(unsigned idx);
protected:

private:
	std::vector<AAttribute *> m_children;
};

class ANumericAttribute : public AAttribute {
public:
	enum NumericAttributeType {
		TUnkownNumeric,
		TByteNumeric,
		TShortNumeric,
		TIntNumeric,
		TFloatNumeric,
		TDoubleNumeric,
		TBooleanNumeric
	};
	
	ANumericAttribute();
	virtual ~ANumericAttribute();
	
	virtual AttributeType attrType() const;
	virtual NumericAttributeType numericType() const;
	
	virtual void setValue(const int & x);
	virtual void setValue(const float & x);
protected:
	
private:
};

class AShortNumericAttribute : public ANumericAttribute {
public:	
	AShortNumericAttribute();
	AShortNumericAttribute(short x);
	virtual ~AShortNumericAttribute();
	
	virtual NumericAttributeType numericType() const;
	
	virtual void setValue(const int & x);
	virtual void setValue(const float & x);
	
	short value() const;
protected:
	
private:
	short m_value;
};

class AByteNumericAttribute : public AShortNumericAttribute {
public:	
	AByteNumericAttribute();
	AByteNumericAttribute(short x);
	virtual ~AByteNumericAttribute();
	
	virtual NumericAttributeType numericType() const;
	
	char asChar() const;
	
protected:
	
private:

};

class AIntNumericAttribute : public ANumericAttribute {
public:	
	AIntNumericAttribute();
	AIntNumericAttribute(int x);
	virtual ~AIntNumericAttribute();
	
	virtual NumericAttributeType numericType() const;
	
	virtual void setValue(const int & x);
	virtual void setValue(const float & x);
	
	int value() const;
protected:
	
private:
	int m_value;
};

class AFloatNumericAttribute : public ANumericAttribute {
public:	
	AFloatNumericAttribute();
	AFloatNumericAttribute(float x);
	virtual ~AFloatNumericAttribute();
	
	virtual NumericAttributeType numericType() const;
	
	virtual void setValue(const int & x);
	virtual void setValue(const float & x);
	
	float value() const;
protected:
	
private:
	float m_value;
};

class ADoubleNumericAttribute : public ANumericAttribute {
public:	
	ADoubleNumericAttribute();
	ADoubleNumericAttribute(double x);
	virtual ~ADoubleNumericAttribute();
	
	virtual NumericAttributeType numericType() const;
	
	virtual void setValue(const int & x);
	virtual void setValue(const float & x);
	
	double value() const;
protected:
	
private:
	double m_value;
};

class ABooleanNumericAttribute : public ANumericAttribute {
public:	
	ABooleanNumericAttribute();
	ABooleanNumericAttribute(bool x);
	virtual ~ABooleanNumericAttribute();
	
	virtual NumericAttributeType numericType() const;
	
	virtual void setValue(const int & x);
	virtual void setValue(const float & x);
	
	bool value() const;
	char asChar() const;
	
protected:
	
private:
	bool m_value;
};
//:~