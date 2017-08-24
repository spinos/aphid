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
	nUnknown = 2048,
	nmNodeName,
	nGrowPortion,
	nGrowMargin,
	nGrowAngle,
	nZenithNoise,
	nFileName,
	nWidth,
	nHeight,
	nRadius,
	nCenterLine,
	nLeftSide,
	nRightSide,
	nBend,
	nTwist,
	nRoll,
	nFold,
	nmCrumple,
	nWidthVariation,
	nHeightVariation,
	nRadiusVariation,
	nSizeVariation,
	nBendVariation,
	nWeightVariation,
	nNoiseVariation,
	nCrumpleVar,
	nFoldVar,
	nAgingVar,
	nPetioleAngle,
	nLeafPlacement,
	nWhorlCount,
	nAddSegment,
	nNProfiles,
	nMidribWidth,
	nMidribThickness,
	nVein,
	nGrowBegin,
	nLengthScale,
	nRadiusScale,
	nNumSeasons,
	nShuffle,
};

enum AttribType {
	tUnknown = 1024,
	tBool,
	tEnum,
	tAction,
	tInt,
	tInt2,
	tInt3,
	tFloat,
	tVec2,
	tVec3,
	tColor,
	tString,
	tSpline,
};

/// monopodial: a single continuous growth axis that 
/// extends at its apex and produces succesive lateral 
/// shoots.
/// sympodial: the apparent main axis or stem of a plant,
/// made up of succesive secondary axes due to the death
/// of each season's terminal bud, as in the vine. 
enum BranchingUnitType {
	buUnknown = 0,
	buMonopodial,
	buSympodial,
};

enum BudType {
	bdUnknown = 768,
	bdTerminal,
	bdLateral,
};

/// phyllotaxy describes the organisation of the leaves on the stem
enum PhyllotaxyValue {
	phOpposite = 512,
	phAlternate,
	phDecussate,
	phWhorled,
};

enum SelectCondition {
	slUnknown = 256,
	slIndex,
	slRandom,
	slAge,
	slCloseToUp,
};

class Attrib {

/// value, value_min, value_max
/// comp0, comp1, comp2
	char m_data[32];
	AttribName m_anm;
	AttribType m_atyp;
	
public:
	Attrib(AttribName anm, AttribType atyp);
	virtual ~Attrib();
	
	AttribName attrName() const;
	AttribType attrType() const;
	
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
	std::string attrNameStr() const;
	
	static gar::AttribName IntAsAttribName(int x);
	static const char* IntAsEnumFieldName(int x);

private:
	static const char* sAttribNameAsStr[];
	
};

class EnumAttrib : public Attrib {

	int * m_fields;
	int m_numFields;
	
public:
	EnumAttrib(AttribName anm);
	virtual ~EnumAttrib();
	
	void createFields(int n);
	void setField(int i, int x);
	
	const int& numFields() const;
	const int& getField(int x) const;
	
};

class SplineAttrib : public Attrib {

	float m_splineValue[2];
/// cv0 cv1
	float m_splineCv[4];
	
public:
	SplineAttrib(AttribName anm);
	virtual ~SplineAttrib();
	
	void setSplineValue(float y0, float y1);
	void setSplineCv0(float x, float y);
	void setSplineCv1(float x, float y);
	
	void getSplineValue(float* y) const;
	void getSplineCv0(float* y) const;
	void getSplineCv1(float* y) const;
	
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

class ActionAttrib : public Attrib {

	std::string m_imgname;
	
public:
	ActionAttrib(AttribName anm);
	virtual ~ActionAttrib();
	
	void setImageName(const std::string& x);
	const std::string& imageName() const;
	
};

}
#endif
