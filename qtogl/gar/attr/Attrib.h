/*
 *  Attrib.h
 *  attribute name type value
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_ATTRIB_H
#define GAR_ATTRIB_H

#include <string>

namespace gar {

enum AttribName {
	nUnknown = 0,
	nGrowMargin,
	nGrowAngle,
	nZenithNoise,
	nFileName
};

enum AttribType {
	tUnknown = 0,
	tBool,
	tInt,
	tFloat,
	tVector2,
	tVector3,
	tColor,
	tString
};

class Attrib {

	AttribName m_anm;
	AttribType m_atyp;
/// value, value_min, value_max
/// comp0, comp1, comp2
	char m_data[24];
	
public:
	Attrib(AttribName anm, AttribType at);
	virtual ~Attrib();
	
	const AttribName& attrName() const;
	const AttribType& attrType() const;
	
	void setValue(const bool& x);
	void setValue(const int& x);
	void setValue(const float& x);
	void setValue2(const int* x);
	void setValue2(const float* x);
	void setValue3(const int* x);
	void setValue3(const float* x);
	
	void setMin(const int& x);
	void setMin(const float& x);
	void setMax(const int& x);
	void setMax(const float& x);
	
	void getValue(bool& y) const;
	void getValue(int& y) const;
	void getValue(float& y) const;
	void getValue2(int* y) const;
	void getValue2(float* y) const;
	void getValue3(int* y) const;
	void getValue3(float* y) const;
	
	void getMin(int& y) const;
	void getMin(float& y) const;
	void getMax(int& y) const;
	void getMax(float& y) const;
	
	bool isStringType() const;
	const char* attrNameStr() const;
	
};

class StringAttrib : public Attrib {

	std::string m_strdata;
	bool m_isFileName;
	
public:
	StringAttrib(AttribName anm, const bool& afn = false);
	virtual ~StringAttrib();
	
	void setValue(const std::string& x);
	void getValue(std::string& y) const;
	
	const bool& isFileName() const;
	
};

}
#endif