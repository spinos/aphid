#include "ConditionNode.h"

#include <maya/MFnMatrixAttribute.h>
#include <maya/MDataHandle.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFloatVector.h>
#include <DynamicsSolver.h>
#include "PhysicsState.h"
namespace caterpillar {

MTypeId ConditionNode::id(0xe85edad);
MObject ConditionNode::a_inDim;
MObject ConditionNode::a_inTime;
MObject ConditionNode::a_outSolver;

ConditionNode::ConditionNode() 
{
}

ConditionNode::~ConditionNode() 
{
}

MStatus ConditionNode::compute( const MPlug& plug, MDataBlock& block )
{		
	if( plug == a_outSolver ) {
		if(PhysicsState::engineStatus == PhysicsState::sCreating) {
			MGlobal::displayInfo("creating condition");
		}
		else if(PhysicsState::engineStatus == PhysicsState::sUpdating) {
			MGlobal::displayInfo("updating condition");
		}
		
		MTime curTime = block.inputValue(a_inTime).asTime();
		
		MDataHandle hdim = block.inputValue(a_inDim);
		MFloatVector &fV = hdim.asFloatVector(); 
		MGlobal::displayInfo(MString("inDIm is ")+ fV.x + " " + fV.y + " " + fV.z);
		
		block.outputValue(a_outSolver).set(true);
        block.setClean(plug);
		return MS::kSuccess;
	}
	return MStatus::kUnknownParameter;
}

void ConditionNode::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{
	view.beginGL();
	glPushAttrib( GL_ALL_ATTRIB_BITS );
    glDisable(GL_LIGHTING);
	glColor3f(1.0, 1.0, 0.0);
	
	glPopAttrib();
	view.endGL();
}

bool ConditionNode::isBounded() const
{ 
	return false;
}

MBoundingBox ConditionNode::boundingBox() const
{   
	
	MPoint corner1(0, 0, 0);
	MPoint corner2(1, 1, 1);

	return MBoundingBox( corner1, corner2 );
}

void* ConditionNode::creator()
{
	return new ConditionNode();
}

MStatus ConditionNode::initialize()
{ 
	MFnNumericAttribute fnNumericAttr;
	MFnUnitAttribute        fnUnitAttr;
	MFnMessageAttribute     fnMsgAttr;
	MStatus			 status;
	
	a_inTime = fnUnitAttr.create( "inTime", "itm", MFnUnitAttribute::kTime, 0.0, &status );
	status = addAttribute(a_inTime);
	
	a_inDim = fnNumericAttr.createPoint("boxDim", "bdm", &status);
    fnNumericAttr.setDefault(1.0, 1.0, 1.0);
    fnNumericAttr.setKeyable(true);
    status = addAttribute(a_inDim);
	
	a_outSolver = fnMsgAttr.create("outSolver", "osv", &status);
    status = addAttribute(a_outSolver);
	
	attributeAffects(a_inTime, a_outSolver);
	
	return MS::kSuccess;
}

}
//:~
