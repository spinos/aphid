#include <maya/MPxTransform.h>
#include <maya/MPxTransformationMatrix.h>
#include <maya/MGlobal.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MIOStream.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include "RigidBodyTransform.h"

namespace caterpillar {

MObject RigidBodyTransformNode::a_inSolver;
MTypeId RigidBodyTransformNode::id(0xdb25c91);
MTypeId RigidBodyTransformMatrix::id(0x22eb5607);

//
// Implementation of our custom transformation matrix
//

//
//	Constructor for matrix
//
RigidBodyTransformMatrix::RigidBodyTransformMatrix()
{
	fm[0][0] =1; fm[0][1] =0; fm[0][2] =0; fm[0][3] =0; 
	fm[1][0] =0; fm[1][1] =1; fm[1][2] =0; fm[1][3] =0; 
	fm[2][0] =0; fm[2][1] =0; fm[2][2] =1; fm[2][3] =0; 
	fm[3][0] =0; fm[3][1] =0; fm[3][2] =0; fm[3][3] =1; 
}

//
// Creator for matrix
//
void *RigidBodyTransformMatrix::creator()
{
	return new RigidBodyTransformMatrix();
}

//
//	Utility method for setting the rcok 
//	motion in the X axis
//
void RigidBodyTransformMatrix::setRockInX( float space[16])
{
	fm[0][0] = space[0];
	fm[0][1] = space[1];
	fm[0][2] = space[2];
	fm[0][3] = space[3];
	fm[1][0] = space[4];
	fm[1][1] = space[5];
	fm[1][2] = space[6];
	fm[1][3] = space[7];
	fm[2][0] = space[8];
	fm[2][1] = space[9];
	fm[2][2] = space[10];
	fm[2][3] = space[11];
	fm[3][0] = space[12];
	fm[3][1] = space[13];
	fm[3][2] = space[14];
	fm[3][3] = space[15];
}

//
// This method will be used to return information to
// Maya.  Use the attributes which are outside of
// the regular transform attributes to build a new
// matrix.  This new matrix will be passed back to
// Maya.
//
MMatrix RigidBodyTransformMatrix::asMatrix() const
{
	// Get the current transform matrix
	MMatrix m = ParentClass::asMatrix();
	// Initialize the new matrix we will calculate
	MTransformationMatrix tm( m );
	// Find the current rotation as a quaternion
	MQuaternion quat = rotation();
	// Convert the rocking value in degrees to radians
	//DegreeRadianConverter conv;
	//double newTheta = conv.degreesToRadians( getRockInX() );
	//quat.setToXAxis( newTheta );
	// Apply the rocking rotation to the existing rotation
	//tm.addRotationQuaternion( quat.x, quat.y, quat.z, quat.w, MSpace::kTransform );
	//MVector tv(0,getRockInX(),0);
	//fm[0][0] =1; fm[0][1] =0; fm[0][2] =0; fm[0][3] =0; 
	//fm[1][0] =0; fm[1][1] =1; fm[1][2] =0; fm[1][3] =0; 
	//fm[2][0] =0; fm[2][1] =0; fm[2][2] =1; fm[2][3] =0; 
	//fm[3][0] =0; fm[3][1] =getRockInX(); fm[3][2] =0; fm[3][3] =1; 
	MMatrix mm(fm);
	//tm.setTranslation( tv,MSpace::kTransform);
	tm = mm;
	// Let Maya know what the matrix should be
	return tm.asMatrix();
}
/*
MMatrix RigidBodyTransformMatrix::asMatrix(double percent) const
{
	MPxTransformationMatrix m(*this);

	//	Apply the percentage to the matrix components
	MVector trans = m.translation();
	trans *= percent;
	m.translateTo( trans );
	MPoint rotatePivotTrans = m.rotatePivot();
	rotatePivotTrans = rotatePivotTrans * percent;
	m.setRotatePivot( rotatePivotTrans );
	MPoint scalePivotTrans = m.scalePivotTranslation();
	scalePivotTrans = scalePivotTrans * percent;
	m.setScalePivotTranslation( scalePivotTrans );

	//	Apply the percentage to the rotate value.  Same
	// as above + the percentage gets applied
	MQuaternion quat = rotation();
	DegreeRadianConverter conv;
	double newTheta = conv.degreesToRadians( getRockInX() );
	quat.setToXAxis( newTheta );
	m.rotateBy( quat );
	MEulerRotation eulRotate = eulerRotation();
	m.rotateTo(  eulRotate * percent, MSpace::kTransform);

	//	Apply the percentage to the scale
	MVector s(scale(MSpace::kTransform));
	s.x = 1.0 + (s.x - 1.0)*percent;
	s.y = 1.0 + (s.y - 1.0)*percent;
	s.z = 1.0 + (s.z - 1.0)*percent;
	m.scaleTo(s, MSpace::kTransform);
    
	return m.asMatrix();
}

MMatrix	RigidBodyTransformMatrix::asRotateMatrix() const
{
	// To be implemented
	return ParentClass::asRotateMatrix();
}
*/

//
// Implementation of our custom transform
//

//
//	Constructor of the transform node
//
RigidBodyTransformNode::RigidBodyTransformNode()
: ParentClass()
{
}

//
//	Constructor of the transform node
//
RigidBodyTransformNode::RigidBodyTransformNode(MPxTransformationMatrix *tm)
: ParentClass(tm)
{
}

//
//	Post constructor method.  Have access to *this.  Node setup
//	operations that do not go into the initialize() method should go
//	here.
//
void RigidBodyTransformNode::postConstructor()
{
	//	Make sure the parent takes care of anything it needs.
	//
	ParentClass::postConstructor();

	// 	The baseTransformationMatrix pointer should be setup properly 
	//	at this point, but just in case, set the value if it is missing.
	//
	if (NULL == baseTransformationMatrix) {
		MGlobal::displayWarning("NULL baseTransformationMatrix found!");
		baseTransformationMatrix = new MPxTransformationMatrix();
	}

	//MPlug aRockInXPlug(thisMObject(), aRockInX);
}

//
//	Destructor of the rocking transform
//
RigidBodyTransformNode::~RigidBodyTransformNode()
{
}

//
//	Method that returns the new transformation matrix
//
MPxTransformationMatrix *RigidBodyTransformNode::createTransformationMatrix()
{
	return new RigidBodyTransformMatrix();
}

//
//	Method that returns a new transform node
//
void *RigidBodyTransformNode::creator()
{
	return new RigidBodyTransformNode();
}

//
//	Node initialize method.  We configure node
//	attributes here.  Static method so
//	*this is not available.
//
MStatus RigidBodyTransformNode::initialize()
{	
	MStatus				status;
	MFnNumericAttribute numAttr;
	MFnUnitAttribute uAttr;
	MFnMessageAttribute     fnMsgAttr;

	a_inSolver = fnMsgAttr.create("inSolver", "isv", &status);
	status = addAttribute(a_inSolver);
	//	This is required so that the validateAndSet method is called
	mustCallValidateAndSet(a_inSolver);
	return MS::kSuccess;
}

//
//	Debugging method
//
const char* RigidBodyTransformNode::className() 
{
	return "caterpillarRigidBody";
}

//
//	Reset transformation
//
void  RigidBodyTransformNode::resetTransformation (const MMatrix &matrix)
{
	ParentClass::resetTransformation( matrix );
}

//
//	Reset transformation
//
void  RigidBodyTransformNode::resetTransformation (MPxTransformationMatrix *resetMatrix )
{
	ParentClass::resetTransformation( resetMatrix );
}

//
// A very simple implementation of validAndSetValue().  No lock
// or limit checking on the rocking attribute is done in this method.
// If you wish to apply locks and limits to the rocking attribute, you
// would follow the approach taken in the RigidBodyTransformCheck example.
// Meaning you would implement methods similar to:
//	* applyRotationLocks();
//	* applyRotationLimits();
//	* checkAndSetRotation();  
// but for the rocking attribute.  The method checkAndSetRotation()
// would be called below rather than updating the rocking attribute
// directly.
//
MStatus RigidBodyTransformNode::validateAndSetValue(const MPlug& plug,
												const MDataHandle& handle,
												const MDGContext& context)
{
	MStatus status = MS::kSuccess;

	//	Make sure that there is something interesting to process.
	//
	if (plug.isNull())
		return MS::kFailure;
		
	MDataBlock block = forceCache(*(MDGContext *)&context);
	//MDataHandle blockHandle = block.outputValue(plug, &status);

	float _tm[16];
		_tm[0] = 1.f; _tm[1] = _tm[2] = _tm[3] = 0.f;
		_tm[4] = 0.f; _tm[5] = 1.f; _tm[6] = _tm[7] = 0.f;
		_tm[8] = _tm[9] = 0.f; _tm[10] = 1.f; _tm[11] = 0.f;
		_tm[12] = _tm[13] = _tm[14] = 0.f; _tm[15] = 1.f;
		
	if ( plug == a_inSolver ) { // MGlobal::displayInfo("tm validate and set insolver");
		block.inputValue(a_inSolver);
		// Update the custom transformation matrix to the
		// right rock value.  
		RigidBodyTransformMatrix *ltm = getRigidBodyTransformMatrix();
		if(ltm)
			ltm->setRockInX(_tm);
		else 
			MGlobal::displayError("Failed to get rock transform matrix");

		// Mark the matrix as dirty so that DG information
		// will update.
		dirtyMatrix();
	}

	// Allow processing for other attributes
	return ParentClass::validateAndSetValue(plug, handle, context);
}

MStatus RigidBodyTransformNode::compute( const MPlug& plug, MDataBlock& data )
{
    MStatus stat; 

	// return MS::kUnknownParameter;

    return MS::kSuccess;
}

//
//	Method for returning the current rocking transformation matrix
//
RigidBodyTransformMatrix *RigidBodyTransformNode::getRigidBodyTransformMatrix()
{
	RigidBodyTransformMatrix *ltm = (RigidBodyTransformMatrix *) baseTransformationMatrix;
	return ltm;
}

}
 