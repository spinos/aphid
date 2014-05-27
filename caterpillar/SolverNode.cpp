#include "SolverNode.h"

#include <maya/MFnMatrixAttribute.h>
#include <maya/MDataHandle.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <DynamicsSolver.h>
namespace caterpillar {

MTypeId SolverNode::id(0x31d84701);
MObject SolverNode::a_inTime;
MObject SolverNode::a_startTime;
MObject SolverNode::a_gravity;
MObject SolverNode::a_enable;
MObject SolverNode::a_numSubsteps;
MObject SolverNode::a_frequency;
MObject SolverNode::a_timeScale;
MObject SolverNode::a_inConditions;
MObject SolverNode::a_outRigidBodies;

SolverNode::SolverNode() 
{
}

SolverNode::~SolverNode() 
{
}

MStatus SolverNode::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == a_outRigidBodies ) {
		bool enabled = block.inputValue(a_enable).asBool();
		
		if(!enabled) {
			// block.outputValue(a_outRigidBodies).set(true);
			// block.setClean(plug);
			return MStatus::kUnknownParameter;
		}
		
		const MTime curTime = block.inputValue(a_inTime).asTime();
		const MTime startTime = block.inputValue(a_startTime).asTime();
		const int numss = block.inputValue(a_numSubsteps).asInt();
		const float freq = block.inputValue(a_frequency).asFloat();
		const float tsl = block.inputValue(a_timeScale).asFloat();
		
		if(curTime == startTime) {
			MGlobal::displayInfo("init solver");
			
			PhysicsState::engine->killPhysics();
			PhysicsState::engine->initPhysics();
			PhysicsState::engineStatus = PhysicsState::sCreating;
			computeConditions(block);
			
		}
		else {
			const double deltaFrame = (curTime - m_preTime).value();
			if(deltaFrame > 0.0 && deltaFrame <= 1.0 && curTime > startTime) {
				if(engine->isWorldInitialized()) {
					PhysicsState::engineStatus = PhysicsState::sUpdating;
					computeConditions(block);
					//MGlobal::displayInfo("sim step");
					const float dt = (float)(curTime - m_preTime).as(MTime::kSeconds) * tsl;
					PhysicsState::engine->simulate(dt, numss, freq);
				}
			}
		}

		m_preTime = curTime;
		
		block.outputValue(a_outRigidBodies).set(curTime.value());
        block.setClean(plug);
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
	
	a_startTime = fnUnitAttr.create( "startTime", "stm", MFnUnitAttribute::kTime, 1.0, &status );
	status = addAttribute(a_startTime);
	
	a_gravity = fnNumericAttr.createPoint("gravity", "grvt", &status);
    fnNumericAttr.setDefault(0.0, -9.81, 0.0);
    status = addAttribute(a_gravity);
	
	a_enable = fnNumericAttr.create("enabled", "enbl", MFnNumericData::kBoolean, true, &status);
    fnNumericAttr.setKeyable(true);
	status = addAttribute(a_enable);
	
	a_numSubsteps = fnNumericAttr.create("substeps", "sbs", MFnNumericData::kInt, 8, &status);
    fnNumericAttr.setKeyable(true);
	fnNumericAttr.setMin(2);
	fnNumericAttr.setMax(100);
    status = addAttribute(a_numSubsteps);

	a_frequency = fnNumericAttr.create("frequency", "fqc", MFnNumericData::kFloat, 90., &status);
    fnNumericAttr.setKeyable(true);
	fnNumericAttr.setMin(60.);
	fnNumericAttr.setMax(6000.);
    status = addAttribute(a_frequency);
	
	a_timeScale = fnNumericAttr.create("timeScale", "tsl", MFnNumericData::kFloat, 1., &status);
    fnNumericAttr.setKeyable(true);
	fnNumericAttr.setMin(0.01);
	fnNumericAttr.setMax(100.);
    status = addAttribute(a_timeScale);
	
	a_inConditions = fnMsgAttr.create("inConditions", "icdts", &status);
	fnMsgAttr.setArray(true);
    status = addAttribute(a_inConditions);
	
	a_outRigidBodies = fnMsgAttr.create("outRigidBodies", "orbds", &status);
    status = addAttribute(a_outRigidBodies);

	attributeAffects(a_inTime, a_outRigidBodies);
	attributeAffects(a_enable, a_outRigidBodies);

	return MS::kSuccess;
}

void SolverNode::computeConditions(MDataBlock& block)
{
	MStatus status;
	MArrayDataHandle hArray = block.inputArrayValue(a_inConditions);
	const int numSlots = hArray.elementCount();
	for(int i=0; i < numSlots; i++) {
		hArray.inputValue(&status);
		hArray.next();
	}
}

}
//:~
