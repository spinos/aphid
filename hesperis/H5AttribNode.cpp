#include "H5AttribNode.h"
#include <SHelper.h>
#include <BaseUtil.h>
#include <boost/format.hpp>

using namespace std;

MTypeId H5AttribNode::id( 0xe04c68c );
MObject H5AttribNode::input;
MObject H5AttribNode::aframe;
MObject H5AttribNode::aminframe;
MObject H5AttribNode::amaxframe;
MObject H5AttribNode::abyteAttrName;
MObject H5AttribNode::outByte;
MObject H5AttribNode::ashortAttrName;
MObject H5AttribNode::outShort;
MObject H5AttribNode::aintAttrName;
MObject H5AttribNode::outInt;
MObject H5AttribNode::afloatAttrName;
MObject H5AttribNode::outFloat;
MObject H5AttribNode::adoubleAttrName;
MObject H5AttribNode::outDouble;
MObject H5AttribNode::aboolAttrName;
MObject H5AttribNode::outBool;
MObject H5AttribNode::aenumAttrName;
MObject H5AttribNode::outEnum;

H5AttribNode::H5AttribNode() {}

H5AttribNode::~H5AttribNode() 
{
	std::map<std::string, aphid::HObject *>::iterator it = m_mappedAttribDatas.begin();
	for(;it!=m_mappedAttribDatas.end();++it) delete it->second;
	m_mappedAttribDatas.clear();
}

MStatus H5AttribNode::compute( const MPlug& plug, MDataBlock& data )
{
	MStatus stat;
	
	MString cacheName = data.inputValue( input ).asString();
	if(cacheName.length() < 2) return stat;
	
    std::string substitutedCacheName(cacheName.asChar());
	EnvVar::replace(substitutedCacheName);

	if(!openH5File(substitutedCacheName) ) {
		aphid::AHelper::Info<std::string >("H5AttribNode cannot open h5 file ", substitutedCacheName );
		return stat;
	}
	
    double dtime = data.inputValue( aframe ).asDouble();
	const int imin = data.inputValue( aminframe ).asInt();
    const int imax = data.inputValue( amaxframe ).asInt();
	if(dtime < imin) dtime = imin;
	if(dtime > imax) dtime = imax;
    
	sampler()->calculateWeights(dtime, sampler()->m_spf);
	sampler()->m_minFrame = imin;

	const unsigned idx = plug.logicalIndex();
	
	if( plug.array() == outByte ) {
		const std::string attrName = getAttrNameInArray(data, abyteAttrName, idx, &stat);
		if(!stat) return MS::kFailure;
		
		char b = 0;
		readData<aphid::HOocArray<aphid::hdata::TChar, 1, 64>, char >(attrName, sampler(), b);
		
		MDataHandle hbt = getHandleInArray(data, outByte, idx, &stat);
		hbt.set(b);
	    
		data.setClean(plug);
	} 
	else if( plug.array() == outShort ) {
		const std::string attrName = getAttrNameInArray(data, ashortAttrName, idx, &stat);
		if(!stat) return MS::kFailure;
		
        short b = 0;
		readData<aphid::HOocArray<aphid::hdata::TShort, 1, 64>, short >(attrName, sampler(), b);
		
		MDataHandle hbt = getHandleInArray(data, outShort, idx, &stat);
		hbt.set(b);
	    
		data.setClean(plug);
	}
	else if( plug.array() == outInt ) {
		const std::string attrName = getAttrNameInArray(data, aintAttrName, idx, &stat);
		if(!stat) return MS::kFailure;
		
		int b = 0;
		readData<aphid::HOocArray<aphid::hdata::TInt, 1, 64>, int >(attrName, sampler(), b);
		
		MDataHandle hbt = getHandleInArray(data, outInt, idx, &stat);
		hbt.set(b);
		
		data.setClean(plug);
	}
	else if( plug.array() == outFloat ) {
		const std::string attrName = getAttrNameInArray(data, afloatAttrName, idx, &stat);
		if(!stat) return MS::kFailure;
		
		float b = 0;
		readData<aphid::HOocArray<aphid::hdata::TFloat, 1, 64>, float >(attrName, sampler(), b);
		
		MDataHandle hbt = getHandleInArray(data, outFloat, idx, &stat);
		hbt.set(b);
	    
		data.setClean(plug);
	}
	else if( plug.array() == outDouble ) {
		const std::string attrName = getAttrNameInArray(data, adoubleAttrName, idx, &stat);
		if(!stat) return MS::kFailure;
		
        double b = 0;
		readData<aphid::HOocArray<aphid::hdata::TDouble, 1, 64>, double >(attrName, sampler(), b);
		
		MDataHandle hbt = getHandleInArray(data, outDouble, idx, &stat);
		hbt.set(b);
	    
		data.setClean(plug);
	}
	else if( plug.array() == outBool ) {
		const std::string attrName = getAttrNameInArray(data, aboolAttrName, idx, &stat);
		if(!stat) return MS::kFailure;
		
		char b = 0;
		readData<aphid::HOocArray<aphid::hdata::TChar, 1, 64>, char >(attrName, sampler(), b);
		
        MDataHandle hbt = getHandleInArray(data, outBool, idx, &stat);
		hbt.set(b);
	    
		data.setClean(plug);
	}
	else if( plug.array() == outEnum ) {
		const std::string attrName = getAttrNameInArray(data, aenumAttrName, idx, &stat);
		if(!stat) return MS::kFailure;
		
		short b = 0;
		readData<aphid::HOocArray<aphid::hdata::TShort, 1, 64>, short >(attrName, sampler(), b);
		
        MDataHandle hbt = getHandleInArray(data, outEnum, idx, &stat);
		hbt.set(b);
	    
		data.setClean(plug);
	} 
	else {
		return MS::kUnknownParameter;
	}

	return MS::kSuccess;
}

void* H5AttribNode::creator()
{
	return new H5AttribNode();
}

MStatus H5AttribNode::initialize()
{
	MFnNumericAttribute numAttr;
	MStatus				stat;
	
	aframe = numAttr.create( "currentTime", "ct", MFnNumericData::kDouble, 1.0 );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute( aframe );
	
	aminframe = numAttr.create( "minFrame", "mnf", MFnNumericData::kInt, 1 );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute( aminframe );
	
	amaxframe = numAttr.create( "maxFrame", "mxf", MFnNumericData::kInt, 24 );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute( amaxframe );
	
	MFnTypedAttribute stringAttr;
	input = stringAttr.create( "cachePath", "cp", MFnData::kString );
 	stringAttr.setStorable(true);
	addAttribute( input );
	
	createNameValueAttr(abyteAttrName, outByte,
						"byteAttribName", "btnm", "outByte", "obt", 
						MFnNumericData::kByte);
	
	createNameValueAttr(ashortAttrName, outShort,
						"shortAttribName", "stnm", "outShort", "ost", 
						MFnNumericData::kShort);
						
	createNameValueAttr(aintAttrName, outInt,
						"intAttribName", "itnm", "outInt", "oit", 
						MFnNumericData::kInt);
	
	createNameValueAttr(afloatAttrName, outFloat,
						"floatAttribName", "ftnm", "outFloat", "oft", 
						MFnNumericData::kFloat);
	
	createNameValueAttr(adoubleAttrName, outDouble,
						"doubleAttribName", "dbnm", "outDouble", "odb", 
						MFnNumericData::kDouble);
	
	createNameValueAttr(aboolAttrName, outBool,
						"boolAttribName", "blnm", "outBool", "obl", 
						MFnNumericData::kBoolean);
/// enum value as short					
	createNameValueAttr(aenumAttrName, outEnum,
						"enumAttribName", "ennm", "outEnum", "oen", 
						MFnNumericData::kShort);

	return MS::kSuccess;
}

void H5AttribNode::createNameValueAttr(MObject & nameAttr, MObject & valueAttr,
						const MString & name1L, const MString & name1S, 
						const MString & name2L, const MString & name2S, 
						MFnNumericData::Type valueTyp)
{
	MFnNumericAttribute numAttr;
	MFnTypedAttribute stringAttr;
	
	nameAttr = stringAttr.create( name1L, name1S, MFnData::kString );
 	stringAttr.setStorable(true);
    stringAttr.setArray(true);
    stringAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( nameAttr );
	
	valueAttr = numAttr.create( name2L, name2S, valueTyp ); 
	numAttr.setStorable(false);
	numAttr.setWritable(false);
    numAttr.setArray(true);
    numAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( valueAttr );
    
	attributeAffects( aframe, valueAttr );
	attributeAffects( input, valueAttr );
}

MStatus H5AttribNode::connectionMade(const MPlug &plug, const MPlug &otherPlug, bool asSrc)
{
    if ( plug.isElement() ) {
        if( plug.array() == outByte) {
 
        }
    }

    return MPxNode::connectionMade( plug, otherPlug, asSrc );
}

std::string H5AttribNode::getAttrNameInArray(MDataBlock& data, const MObject & attr, 
												unsigned idx, MStatus * stat) const
{
	MArrayDataHandle nameArray = data.inputArrayValue( attr, stat );
	nameArray.jumpToElement(idx);
	const MString sname = nameArray.inputValue().asString();
	
	if(!aphid::HObject::FileIO.checkExist(sname.asChar() ) ) {
		aphid::AHelper::Info<MString >("H5AttribNode cannot find grp ", sname );
		*stat = MS::kFailure;
		return "";
	}

	*stat = MS::kSuccess;
	return std::string(sname.asChar() );
}

MDataHandle H5AttribNode::getHandleInArray(MDataBlock& data, const MObject & attr, 
						unsigned idx, MStatus * stat) const
{
	MArrayDataHandle btArry = data.outputArrayValue(attr, stat);
	btArry.jumpToElement(idx);
	return btArry.outputValue();
}
//:~
