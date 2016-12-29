/*
 *  AttributeHelper.h
 *  opium
 *
 *  Created by jian zhang on 9/19/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MAMA_ATTRIBUTE_HELPER_H
#define APH_MAMA_ATTRIBUTE_HELPER_H

#include <foundation/AAttribute.h>
#include <maya/MFnNumericData.h>
#include <maya/MGlobal.h>
#include <maya/MPlug.h>
#include <maya/MDataHandle.h>
#include <boost/lexical_cast.hpp>

class MObject;
class MFnDependencyNode;
class MDGContext;
class MString;

namespace aphid {

class AttributeHelper {
public:
	AttributeHelper();
	
	template<typename Td, typename Tf>
	static bool SaveArrayDataPlug(const Td & dataArray, MPlug & plug)
	{
		Tf dataFn;
		MStatus stat;
		MObject od = dataFn.create(dataArray, &stat);
		if(!stat) {
			return false;
		}
		plug.setValue(od);
		return true;
	}
	
	template<typename Td, typename Tf>
	static bool LoadArrayDataPlug(Td & dataArray, const MPlug & plug)
	{
		MObject od;
		MStatus stat;
		stat = plug.getValue(od);
		if(!stat) {
			return false;
		}
		Tf dataFn(od, &stat);
		if(!stat) {
			return false;
		}
		
		dataArray = dataFn.array();
		return true;
	}
	
	template<typename Td, typename Tf>
	static bool LoadArrayDataHandle(Td & dataArray, MDataHandle & dataH)
	{
		MStatus stat;
		Tf dataFn(dataH.data(), &stat);
		if(!stat) {
			return false;
		}
		
		dataArray = dataFn.array();
		return true;
	}
	
	static void getColorAttributeByName(const MFnDependencyNode& fnode, const char* attrname, double& r, double& g, double& b);
	static void getNormalAttributeByName(const MFnDependencyNode& fnode, const char* attrname, double& r, double& g, double& b);
	static char getDoubleAttributeByName(const MFnDependencyNode& fnode, const char* attrname, double& v);
	static char getBoolAttributeByName(const MFnDependencyNode& fnode, const char* attrname, bool& v);
	static char getDoubleAttributeByNameAndTime(const MFnDependencyNode& fnode, const char* attrname, MDGContext & ctx, double& v);
	static char getStringAttributeByName(const MFnDependencyNode& fnode, const char* attrname, MString& v);
	static char getStringAttributeByName(const MObject& node, const char* attrname, MString& v);
	
	static void setCustomStringAttrib(MObject node, const char* nameLong, const char* nameShort, const char* value);
	
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
	static bool HasNamedAttribute(MObject & attrib, const MObject & node, const std::string & name);
	static bool IsStringAttr(MObject & entity);
	static bool IsEnumAttr(MObject & entity);
	static bool IsNumericAttr(MObject & entity, MFnNumericData::Type t);
	static bool AddStringAttr(MObject & attr, 
	                        const MObject & node, 
							const std::string & nameLong, 
							const std::string & nameShort);
	static bool AddEnumAttr(MObject & attr, 
	                        const MObject & node, 
							const std::string & nameLong, 
							const std::string & nameShort,
							const std::map<short, std::string > & fields);
	static bool AddNumericAttr(MObject & attr, 
	                        const MObject & node, 
							const std::string & nameLong, 
							const std::string & nameShort,
							MFnNumericData::Type t);
	static bool IsDirectAnimated(const MPlug & attrib);
	
};

}
#endif