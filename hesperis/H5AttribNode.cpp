#include <AHelper.h>
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
MObject H5AttribNode::alongAttrName;
MObject H5AttribNode::outLong;
MObject H5AttribNode::afloatAttrName;
MObject H5AttribNode::outFloat;
MObject H5AttribNode::adoubleAttrName;
MObject H5AttribNode::outDouble;
MObject H5AttribNode::aboolAttrName;
MObject H5AttribNode::outBool;

H5AttribNode::H5AttribNode() {}

H5AttribNode::~H5AttribNode() {}

MStatus H5AttribNode::compute( const MPlug& plug, MDataBlock& data )
{
	MStatus stat;
	
	MString cacheName = data.inputValue( input ).asString();
	if(cacheName.length() < 2) return stat;
	
	std::string substitutedCacheName(cacheName.asChar());
	EnvVar::replace(substitutedCacheName);
	
	if(!openH5File(substitutedCacheName) ) {
		AHelper::Info<std::string >("H5AttribNode cannot open h5 file ", substitutedCacheName );
		return stat;
	}
	
	double dtime = data.inputValue( aframe ).asDouble();
	const int imin = data.inputValue( aminframe ).asInt();
    const int imax = data.inputValue( amaxframe ).asInt();
	if(dtime < imin) dtime = imin;
	if(dtime > imax) dtime = imax;
    
	SampleFrame sampler;
	getSampler(sampler);
	sampler.calculateWeights(dtime);

	if( plug.array() == outByte ) {
		const unsigned idx = plug.logicalIndex();
        
		MArrayDataHandle btNameArray = data.inputArrayValue( abyteAttrName );
		btNameArray.jumpToElement(idx);
		const MString btName = btNameArray.inputValue().asString();
		
		if(!HObject::FileIO.checkExist(btName.asChar() ) ) {
			AHelper::Info<MString >("H5AttribNode cannot find grp ", btName );
			return stat;
		}
	
		
		
        MArrayDataHandle btArry = data.outputArrayValue(outByte, &stat);
		btArry.jumpToElement(idx);
		MDataHandle hbt = btArry.outputValue();

		hbt.set(0);
	    
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
	
	abyteAttrName = stringAttr.create( "byteAttribName", "btnm", MFnData::kString );
 	stringAttr.setStorable(true);
    stringAttr.setArray(true);
    stringAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( abyteAttrName );
	
	outByte = numAttr.create( "outByte", "obt", MFnNumericData::kByte ); 
	numAttr.setStorable(false);
	numAttr.setWritable(false);
    numAttr.setArray(true);
    numAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( outByte );
    
	attributeAffects( aframe, outByte );
	attributeAffects( input, outByte );
	
	return MS::kSuccess;
}

MStatus H5AttribNode::connectionMade(const MPlug &plug, const MPlug &otherPlug, bool asSrc)
{
    if ( plug.isElement() ) {
        if( plug.array() == outByte) {
 
        }
    }

    return MPxNode::connectionMade( plug, otherPlug, asSrc );
}
//:~
