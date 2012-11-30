#include "recordNode.h"
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnMatrixAttribute.h>

MTypeId     PoseRecordNode::id( 0x9c0a5b );
MObject PoseRecordNode::aposespace;
MObject     PoseRecordNode::output;       

PoseRecordNode::PoseRecordNode() {}
PoseRecordNode::~PoseRecordNode() {}

MStatus PoseRecordNode::compute( const MPlug& plug, MDataBlock& data )
{
	
	MStatus returnStatus;
 
	if( plug == output )
	{
		float result = 1.0f;
		MDataHandle outputHandle = data.outputValue( PoseRecordNode::output );
		outputHandle.set( result );
		data.setClean(plug);
		
	} else {
		return MS::kUnknownParameter;
	}

	return MS::kSuccess;
}

void* PoseRecordNode::creator()
{
	return new PoseRecordNode();
}

MStatus PoseRecordNode::initialize()
{
	MFnNumericAttribute numAttr;
	MFnTypedAttribute tAttr;
	MStatus				stat;
	
	MFnMatrixAttribute matAttr;
	aposespace = matAttr.create( "poseSpace", "pssp", MFnMatrixAttribute::kDouble );
 	matAttr.setStorable(true);
	matAttr.setArray(true);
	matAttr.setUsesArrayDataBuilder(true); 
	addAttribute(aposespace);
	
	output = numAttr.create( "output", "out", MFnNumericData::kFloat, 0.0 );
	numAttr.setWritable(false);
	numAttr.setStorable(false);
	stat = addAttribute( output );
		if (!stat) { stat.perror("addAttribute"); return stat;}
		
	attributeAffects(aposespace, output);

	return MS::kSuccess;
}
