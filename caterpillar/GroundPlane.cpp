#include "GroundPlane.h"

#include <maya/MFnMatrixAttribute.h>
#include <maya/MDataHandle.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFloatVector.h>
#include <maya/MFnMesh.h>
#include <maya/MFnMeshData.h>
#include <maya/MPointArray.h>
#include <maya/MFloatVectorArray.h>
#include <DynamicsSolver.h>
#include "PhysicsState.h"
namespace caterpillar {

MTypeId GroundPlane::id(0x170d562d);
MObject GroundPlane::a_inMesh;
MObject GroundPlane::a_inTime;
MObject GroundPlane::a_inFriction;
MObject GroundPlane::a_inMargin;
MObject GroundPlane::a_outSolver;

GroundPlane::GroundPlane() 
{
}

GroundPlane::~GroundPlane() 
{
}

MStatus GroundPlane::compute( const MPlug& plug, MDataBlock& block )
{		
	if( plug == a_outSolver ) {
		if(PhysicsState::engineStatus == PhysicsState::sCreating) {
			computeCreate(block);
		}
		else if(PhysicsState::engineStatus == PhysicsState::sUpdating) {
			computeUpdate(block);
		}
		
		MTime curTime = block.inputValue(a_inTime).asTime();
		
		// MGlobal::displayInfo(MString("inDIm is ")+ fV.x + " " + fV.y + " " + fV.z);
		
		block.outputValue(a_outSolver).set(true);
        block.setClean(plug);
		return MS::kSuccess;
	}
	return MStatus::kUnknownParameter;
}

void GroundPlane::draw( M3dView & view, const MDagPath & path, 
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

bool GroundPlane::isBounded() const
{ 
	return false;
}

MBoundingBox GroundPlane::boundingBox() const
{   
	
	MPoint corner1(0, 0, 0);
	MPoint corner2(1, 1, 1);

	return MBoundingBox( corner1, corner2 );
}

void* GroundPlane::creator()
{
	return new GroundPlane();
}

MStatus GroundPlane::initialize()
{ 
	MFnNumericAttribute fnNumericAttr;
	MFnUnitAttribute        fnUnitAttr;
	MFnMessageAttribute     fnMsgAttr;
	MFnTypedAttribute typedAttrFn;
	MStatus			 status;
	
	a_inTime = fnUnitAttr.create( "inTime", "itm", MFnUnitAttribute::kTime, 0.0, &status );
	status = addAttribute(a_inTime);
	
	a_inMesh = typedAttrFn.create("inMesh", "ims", MFnMeshData::kMesh, &status);
	typedAttrFn.setStorable(false);
    status = addAttribute(a_inMesh);
	
	a_outSolver = fnMsgAttr.create("outSolver", "osv", &status);
    status = addAttribute(a_outSolver);
	
	a_inMargin = fnNumericAttr.create("collisionMargin", "clmg", MFnNumericData::kFloat, 1.f, &status);
	fnNumericAttr.setDefault(1.f);
	fnNumericAttr.setMin(0.1f);
	fnNumericAttr.setKeyable(true);
	status = addAttribute(a_inMargin);
	
	a_inFriction = fnNumericAttr.create("friction", "frct", MFnNumericData::kFloat, .732f, &status);
	fnNumericAttr.setDefault(.732f);
	fnNumericAttr.setKeyable(true);
	fnNumericAttr.setMin(0.f);
	status = addAttribute(a_inFriction);
	
	attributeAffects(a_inTime, a_outSolver);
	
	return MS::kSuccess;
}

void GroundPlane::computeCreate(MDataBlock& block)
{
	MObject mesh = block.inputValue(a_inMesh).asMesh();
	if(mesh == MObject::kNullObj) {
		MGlobal::displayWarning("ground plane has no mesh");
		return;
	}
	MStatus status;
	MFnMesh fmesh(mesh, &status);
	if(!status) {
	    MGlobal::displayWarning("ground plane cannot create mesh fn");
		return;
	}
	MIntArray triangleCounts;
	MIntArray triangleVertices;
	fmesh.getTriangles(triangleCounts, triangleVertices);
	int numTri = 0;
	int i, j;
	for(i = 0; i < triangleCounts.length(); i++)
	    numTri += triangleCounts[i];
	
	MGlobal::displayInfo(MString("ground plane triangle count: ")+numTri);
	
	int * indices = createTriangles(numTri);
	
	j = 0;
	for(i = 0; i < triangleVertices.length(); i++) {
	    indices[j] = triangleVertices[i];
	    j++;
	}
	MPointArray vertexArray;
	fmesh.getPoints(vertexArray);
	const int nv = vertexArray.length();
	MGlobal::displayInfo(MString("ground plane vertex count: ")+nv);
	
	MFloatVectorArray normalArray;
	fmesh.getNormals(normalArray);
	
	const float fmargin = block.inputValue(a_inMargin).asFloat();
	
	btVector3 * vs = createVertexPos(nv);
	
	for(i=0; i < nv; i++) {
	    const MPoint & q = vertexArray[i];
	    vs[i][0] = q.x - normalArray[i].x * fmargin;
		vs[i][1] = q.y - normalArray[i].y * fmargin;
		vs[i][2] = q.z - normalArray[i].z * fmargin;
	}
	
	setMargin(fmargin);
	const float friction = block.inputValue(a_inFriction).asFloat();
	setFriction(friction);
	create();
	
	triangleCounts.clear();
	triangleVertices.clear();
}

void GroundPlane::computeUpdate(MDataBlock& data)
{

}

}
//:~
