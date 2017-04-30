#include "recordNode.h"
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MVectorArray.h>

MTypeId     PoseRecordNode::id( 0x9c0a5b );
MObject PoseRecordNode::aposespacerow0;
MObject PoseRecordNode::aposespacerow1;
MObject PoseRecordNode::aposespacerow2;
MObject PoseRecordNode::aposespacerow3;
MObject PoseRecordNode::abindpnt;
MObject PoseRecordNode::aposepnt;
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
	MStatus stat;
	MVectorArray defaultVectArray;
	MFnVectorArrayData vectArrayDataFn;
	vectArrayDataFn.create( defaultVectArray );
	
	aposespacerow0 = tAttr.create( "poseSpaceRow0", "psr0", MFnData::kVectorArray, vectArrayDataFn.object());
 	tAttr.setStorable(true);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aposespacerow0);
	
	aposespacerow1 = tAttr.create( "poseSpaceRow1", "psr1", MFnData::kVectorArray, vectArrayDataFn.object());
 	tAttr.setStorable(true);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aposespacerow1);
	
	aposespacerow2 = tAttr.create( "poseSpaceRow2", "psr2", MFnData::kVectorArray, vectArrayDataFn.object());
 	tAttr.setStorable(true);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aposespacerow2);
	
	aposespacerow3 = tAttr.create( "poseSpaceRow3", "psr3", MFnData::kVectorArray, vectArrayDataFn.object());
 	tAttr.setStorable(true);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aposespacerow3);
	
	abindpnt = tAttr.create( "bindPoint", "bpnt", MFnData::kVectorArray, vectArrayDataFn.object());
 	tAttr.setStorable(true);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(abindpnt);
	
	aposepnt = tAttr.create( "posePoint", "ppnt", MFnData::kVectorArray, vectArrayDataFn.object());
 	tAttr.setStorable(true);
 	tAttr.setWritable(true);
 	tAttr.setReadable(true);
	addAttribute(aposepnt);

	output = numAttr.create( "output", "out", MFnNumericData::kFloat, 0.0 );
	numAttr.setWritable(false);
	numAttr.setStorable(false);
	stat = addAttribute( output );
		if (!stat) { stat.perror("addAttribute"); return stat;}
		
	attributeAffects(aposespacerow0, output);
	attributeAffects(aposespacerow1, output);
	attributeAffects(aposespacerow2, output);
	attributeAffects(aposespacerow3, output);
	attributeAffects(abindpnt, output);
	attributeAffects(aposepnt, output);

	return MS::kSuccess;
}
