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
MObject     BoundTranslateNode::compoundOutput;      
MObject		BoundTranslateNode::constraintTranslateX;
MObject		BoundTranslateNode::constraintTranslateY;
MObject		BoundTranslateNode::constraintTranslateZ;

MObject BoundTranslateNode::atargetBound;

BoundTranslateNode::BoundTranslateNode() 
{}

BoundTranslateNode::~BoundTranslateNode() 
{}

void BoundTranslateNode::postConstructor()
{}

MStatus BoundTranslateNode::compute( const MPlug& plug, MDataBlock& block )
{	
	MStatus stat;
	if(!m_isInitd) return stat;
    
    if(plug == constraintTranslateX || 
        plug == constraintTranslateY ||
        plug == constraintTranslateZ) {
         // AHelper::Info<MString>("ov child", plug.name());
         // AHelper::Info<unsigned>("ov id", plug.parent().logicalIndex());
         unsigned iobject = plug.parent().logicalIndex();
         if(iobject > m_numObjects-1) {
             MGlobal::displayInfo("n constraint is out of bound");
             return MS::kSuccess;
         }
         
         if(iobject == 0 && plug == constraintRotateX) {
             MDataHandle hm = block.inputValue(atargetMesh);
             updateShape(hm.asMesh());
         }
		       
         MDataHandle hout = block.outputValue(plug, &stat);
             
         if(plug == constraintTranslateX) {
             hout.set(m_solvedT.x);
         }
         else if(plug == constraintTranslateY) {
             hout.set(m_solvedT.y);
         }
         else if(plug == constraintTranslateZ) {
             hout.set(m_solvedT.z);
         }
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
	MFnNumericAttribute nAttr;
	MStatus				status;

	MFnTypedAttribute typedAttr;
    
    MFnNumericAttribute numAttr;
    constraintTranslateX = numAttr.create( "constraintTranslateX", "ctx", MFnNumericData::kDouble, 0.0, &status );
    if(!status) {
        MGlobal::displayInfo("failed to create attrib constraintTranslateX");
        return status;
    }
    
    constraintTranslateY = numAttr.create( "constraintTranslateY", "cty", MFnNumericData::kDouble, 0.0, &status );
    if(!status) {
        MGlobal::displayInfo("failed to create attrib constraintTranslateY");
        return status;
    }
    
    constraintTranslateZ = numAttr.create( "constraintTranslateZ", "ctz", MFnNumericData::kDouble, 0.0, &status );
    if(!status) {
        MGlobal::displayInfo("failed to create attrib constraintTranslateY");
        return status;
    }
    
	MFnCompoundAttribute compoundAttr;
	compoundOutput = compoundAttr.create( "outValue", "otv",&status );
	if (!status) { status.perror("compoundAttr.create"); return status;}
	status = compoundAttr.addChild( constraintTranslateX );
	if (!status) { status.perror("compoundAttr.addChild tx"); return status;}
	status = compoundAttr.addChild( constraintTranslateY );
	if (!status) { status.perror("compoundAttr.addChild ty"); return status;}
	status = compoundAttr.addChild( constraintTranslateZ );
	if (!status) { status.perror("compoundAttr.addChild tz"); return status;}
	compoundAttr.addChild( constraintRotateX );
	compoundAttr.addChild( constraintRotateY );
	compoundAttr.addChild( constraintRotateZ );

	status = addAttribute( compoundOutput );
	if (!status) { status.perror("addAttribute"); return status;}

	atargetBound = typedAttr.create("targetMesh", "tgms", MFnMeshData::kMesh, &status);
	typedAttr.setStorable(false);
	addAttribute(atargetBound);
	
    attributeAffects(atargetBound, compoundOutput);
    
	return MS::kSuccess;
}
//:~
