#include "BoundTranslateNode.h"
#include <maya/MFnMatrixData.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnMeshData.h>
#include <maya/MFnMesh.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MTransformationMatrix.h>
#include <AHelper.h>

MTypeId     BoundTranslateNode::id( 0xa2f8bf );
//MObject     BoundTranslateNode::compoundOutput;      
MObject		BoundTranslateNode::constraintTranslateX;
MObject		BoundTranslateNode::constraintTranslateY;
MObject		BoundTranslateNode::constraintTranslateZ;

//MObject BoundTranslateNode::ainBoundMin;
MObject BoundTranslateNode::ainBoundMinX;
MObject BoundTranslateNode::ainBoundMinY;
MObject BoundTranslateNode::ainBoundMinZ;
//MObject BoundTranslateNode::ainBoundMax;
MObject BoundTranslateNode::ainBoundMaxX;
MObject BoundTranslateNode::ainBoundMaxY;
MObject BoundTranslateNode::ainBoundMaxZ;

BoundTranslateNode::BoundTranslateNode() 
{}

BoundTranslateNode::~BoundTranslateNode() 
{}

void BoundTranslateNode::postConstructor()
{}

MStatus BoundTranslateNode::compute( const MPlug& plug, MDataBlock& block )
{	
	MStatus stat;
    
    if(plug == constraintTranslateX ) {
        computeBoundCenter(block);
        MDataHandle hout = block.outputValue(constraintTranslateX);
        hout.set(m_boundCenter.x);
        block.setClean( plug );
    }
    else if(plug == constraintTranslateY) {
        computeBoundCenter(block);
        MDataHandle hout = block.outputValue(constraintTranslateY);
        hout.set(m_boundCenter.y);
        block.setClean( plug );
    }
    else if(plug == constraintTranslateZ) {
        computeBoundCenter(block);
        MDataHandle hout = block.outputValue(constraintTranslateZ);
        hout.set(m_boundCenter.z);
        block.setClean( plug ); 
    }
	else
		return MS::kUnknownParameter;

	return MS::kSuccess;
}

void* BoundTranslateNode::creator()
{
	return new BoundTranslateNode;
}

MStatus BoundTranslateNode::initialize()
{
	MStatus				status;

	MFnTypedAttribute typedAttr;
    
    MFnNumericAttribute numAttr;

    ainBoundMinX = numAttr.create( "bBoxMinX", "bbmnx", MFnNumericData::kDouble, 0.0, &status );
    ainBoundMinY = numAttr.create( "bBoxMinY", "bbmny", MFnNumericData::kDouble, 0.0, &status );
    ainBoundMinZ = numAttr.create( "bBoxMinZ", "bbmnz", MFnNumericData::kDouble, 0.0, &status );
    addAttribute(ainBoundMinX);
    addAttribute(ainBoundMinY);
    addAttribute(ainBoundMinZ);
    /*
    MFnCompoundAttribute compoundAttr;
    ainBoundMin = compoundAttr.create( "inBoundingBoxMin", "bbmn", &status );
    compoundAttr.addChild( ainBoundMinX );
    compoundAttr.addChild( ainBoundMinY );
    compoundAttr.addChild( ainBoundMinZ );
    addAttribute(ainBoundMin);*/

    ainBoundMaxX = numAttr.create( "bBoxMaxX", "bbmxx", MFnNumericData::kDouble, 0.0, &status );
    ainBoundMaxY = numAttr.create( "bBoxMaxY", "bbmxy", MFnNumericData::kDouble, 0.0, &status );
    ainBoundMaxZ = numAttr.create( "bBoxMaxZ", "bbmxz", MFnNumericData::kDouble, 0.0, &status );
    addAttribute(ainBoundMaxX);
    addAttribute(ainBoundMaxY);
    addAttribute(ainBoundMaxZ);
    /*ainBoundMax = compoundAttr.create( "inBoundingBoxMax", "bbmx", &status );
    compoundAttr.addChild( ainBoundMaxX );
    compoundAttr.addChild( ainBoundMaxY );
    compoundAttr.addChild( ainBoundMaxZ );
    addAttribute(ainBoundMax);*/
 
    constraintTranslateX = numAttr.create( "outTranslateX", "ctx", MFnNumericData::kDouble, 0.0, &status );
    numAttr.setWritable(false);
    numAttr.setStorable(false);
    if(!status) {
        MGlobal::displayInfo("failed to create attrib constraintTranslateX");
        return status;
    }
    addAttribute(constraintTranslateX);
    
    constraintTranslateY = numAttr.create( "outTranslateY", "cty", MFnNumericData::kDouble, 0.0, &status );;
    numAttr.setWritable(false);
    numAttr.setStorable(false);
    if(!status) {
        MGlobal::displayInfo("failed to create attrib constraintTranslateY");
        return status;
    }
    addAttribute(constraintTranslateY);
    
    constraintTranslateZ = numAttr.create( "outTranslateZ", "ctz", MFnNumericData::kDouble, 0.0, &status );
    numAttr.setWritable(false);
    numAttr.setStorable(false);
    if(!status) {
        MGlobal::displayInfo("failed to create attrib constraintTranslateY");
        return status;
    }
    addAttribute(constraintTranslateZ);
    
	/*compoundOutput = compoundAttr.create( "outValue", "otv",&status );
	if (!status) { status.perror("compoundAttr.create"); return status;}
	status = compoundAttr.addChild( constraintTranslateX );
	if (!status) { status.perror("compoundAttr.addChild tx"); return status;}
	status = compoundAttr.addChild( constraintTranslateY );
	if (!status) { status.perror("compoundAttr.addChild ty"); return status;}
	status = compoundAttr.addChild( constraintTranslateZ );
	if (!status) { status.perror("compoundAttr.addChild tz"); return status;}

	status = addAttribute( compoundOutput );
	if (!status) { status.perror("addAttribute"); return status;}
*/
/*
    attributeAffects(ainBoundMinX, constraintTranslateX);
    attributeAffects(ainBoundMinX, constraintTranslateY);
    attributeAffects(ainBoundMinX, constraintTranslateZ);
    attributeAffects(ainBoundMaxX, constraintTranslateX);
    attributeAffects(ainBoundMaxX, constraintTranslateY);
    attributeAffects(ainBoundMaxX, constraintTranslateZ);*/
    attributeAffects(ainBoundMinX, constraintTranslateX);
    attributeAffects(ainBoundMinX, constraintTranslateY);
    attributeAffects(ainBoundMinX, constraintTranslateZ);
    attributeAffects(ainBoundMinX, constraintTranslateX);
    attributeAffects(ainBoundMinX, constraintTranslateY);
    attributeAffects(ainBoundMinX, constraintTranslateZ);
	return MS::kSuccess;
}

void BoundTranslateNode::computeBoundCenter(MDataBlock& block)
{
    double mnx = block.inputValue(ainBoundMinX).asDouble();
    double mny = block.inputValue(ainBoundMinY).asDouble();
    double mnz = block.inputValue(ainBoundMinZ).asDouble();
    double mxx = block.inputValue(ainBoundMaxX).asDouble();
    double mxy = block.inputValue(ainBoundMaxY).asDouble();
    double mxz = block.inputValue(ainBoundMaxZ).asDouble();
    
    m_boundCenter.x = (mnx + mxx) * .5;
    m_boundCenter.y = (mny + mxy) * .5;
    m_boundCenter.z = (mnz + mxz) * .5;
}
//:~
