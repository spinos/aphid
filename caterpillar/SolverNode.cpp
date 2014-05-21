#include <Vector2F.h>
#include "SolverNode.h"
#include <maya/MString.h> 
#include <maya/MGlobal.h>

#include <maya/MVector.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MDataHandle.h>
#include <maya/MColor.h>
#include <maya/MDistance.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MVectorArray.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MEulerRotation.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MIntArray.h>
#include <maya/MPointArray.h>
#include <maya/MFnMeshData.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MFnMessageAttribute.h>
#include <fstream> 
#include <DynamicsSolver.h>
namespace caterpillar {

MTypeId SolverNode::id(0x31d84701);
MObject SolverNode::a_inTime;
MObject SolverNode::a_startTime;
MObject SolverNode::a_gravity;
MObject SolverNode::a_enable;
MObject SolverNode::a_numSubsteps;
MObject SolverNode::a_frequency;
MObject SolverNode::a_outRigidBodies;
DynamicsSolver * SolverNode::engine = NULL;

SolverNode::SolverNode() 
{
	engine = new DynamicsSolver;
}

SolverNode::~SolverNode() 
{
	delete engine;
}

MStatus SolverNode::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == a_outRigidBodies ) {
			return MS::kSuccess;
	}
	return MStatus::kUnknownParameter;
}

void SolverNode::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{
	view.beginGL();
	glPushAttrib( GL_ALL_ATTRIB_BITS );
    glDisable(GL_LIGHTING);
	glColor3f(1.0, 1.0, 0.0);
	
	engine->renderWorld();
	
	glPopAttrib();
	view.endGL();
}

bool SolverNode::isBounded() const
{ 
	return false;
}

MBoundingBox SolverNode::boundingBox() const
{   
	
	MPoint corner1(0, 0, 0);
	MPoint corner2(1, 1, 1);

	return MBoundingBox( corner1, corner2 );
}

void* SolverNode::creator()
{
	return new SolverNode();
}

MStatus SolverNode::initialize()
{ 
	MFnNumericAttribute fnNumericAttr;
	MFnUnitAttribute        fnUnitAttr;
	MFnMessageAttribute     fnMsgAttr;
	MStatus			 status;
	
	a_inTime = fnUnitAttr.create( "inTime", "itm", MFnUnitAttribute::kTime, 0.0, &status );
	status = addAttribute(a_inTime);
	
	a_inTime = fnUnitAttr.create( "startTime", "stm", MFnUnitAttribute::kTime, 1.0, &status );
	status = addAttribute(a_inTime);
	
	a_gravity = fnNumericAttr.createPoint("gravity", "grvt", &status);
    fnNumericAttr.setDefault(0.0, -9.81, 0.0);
    fnNumericAttr.setKeyable(true);
    status = addAttribute(a_gravity);
	
	a_enable = fnNumericAttr.create("enabled", "enbl", MFnNumericData::kBoolean, true, &status);
    status = addAttribute(a_enable);
	
	a_numSubsteps = fnNumericAttr.create("substeps", "sbs", MFnNumericData::kInt, 1, &status);
    fnNumericAttr.setKeyable(true);
	fnNumericAttr.setMin(1);
	fnNumericAttr.setMax(100);
    status = addAttribute(a_numSubsteps);

	a_frequency = fnNumericAttr.create("frequency", "fqc", MFnNumericData::kInt, 240, &status); //MB
    fnNumericAttr.setKeyable(true);
	fnNumericAttr.setMin(60);
	fnNumericAttr.setMax(6000);
    status = addAttribute(a_frequency);
	
	a_outRigidBodies = fnMsgAttr.create("outRigidBodies", "orbds", &status);
    status = addAttribute(a_outRigidBodies);

	attributeAffects(a_inTime, a_outRigidBodies);
	attributeAffects(a_enable, a_outRigidBodies);
	return MS::kSuccess;
}

}
//:~
