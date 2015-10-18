/*
 *  aAttributeHelper.h
 *  opium
 *
 *  Created by jian zhang on 9/19/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AHelper.h"
#include <AAttribute.h>
class MString;
class MObject;

class AAttributeHelper : public AHelper {
public:
	AAttributeHelper() {}
	static void setCustomStringAttrib(MObject node, const char* nameLong, const char* nameShort, const char* value);
	static char getStringAttrib(MObject node, const char* nameLong, MString& value);
	
	template<typename T>
	static T GetPlugValue(const MPlug & pl)
	{
		T val;
		pl.getValue(val);
		return val;
	}
	
	template<typename T>
	static std::string GetPlugValueAsStr(const MPlug & pl)
	{
		T val;
		pl.getValue(val);
		std::stringstream sst;
		sst<<val;
		return sst.str();
	}
	
	template<typename T>
	static void SetPlugValueFromStr(MPlug & plg, const std::string & src)
	{
		T val = boost::lexical_cast<T>(src);
		MStatus stat = plg.setValue(val);
		if(!stat) MGlobal::displayInfo("failed to set value "+plg.partialName());
	}
	
	static AAttribute::AttributeType GetAttribType(const MObject & entity);
	static MFnNumericData::Type GetNumericAttributeType(const std::string & name);
	
	static AStringAttribute * AsStrData(const MPlug & plg);
	static ANumericAttribute * AsNumericData(const MPlug & plg);
	static ACompoundAttribute * AsCompoundData(const MPlug & plg);
	static AEnumAttribute * AsEnumData(const MPlug & plg);
	static AAttribute::AttributeType AsAttributeType(const std::string & name);
};