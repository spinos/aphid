#include "zsoftIkSolver.h"
#include <maya/MFnNumericAttribute.h>

MTypeId ik2Bsolver::id(0x8fa7476);

MObject ik2Bsolver::asoftDistance;
MObject ik2Bsolver::arestLength1;
MObject ik2Bsolver::arestLength2;
MObject ik2Bsolver::amaxStretching;

ik2Bsolver::ik2Bsolver()
        : MPxIkSolverNode()
{
    m_herm._P[0].set(0.f, 0.f, 0.f);
    m_herm._P[1].set(1.f, 1.f, 0.f);
    m_herm._T[0].set(1.414f, 0.f, 0.f);
    m_herm._T[1].set(1.f, 1.f, 0.f);
}

ik2Bsolver::~ik2Bsolver() {}

void ik2Bsolver::postConstructor()
{
        setRotatePlane(true);
}

void* ik2Bsolver::creator()
{
        return new ik2Bsolver;
}

MStatus ik2Bsolver::initialize()
{ 
    MFnNumericAttribute numattr;
	
	arestLength1 = numattr.create("restLength1", "rsl1", MFnNumericData::kDouble, 16.0);
	numattr.setMin(1.0);
	numattr.setKeyable(true); 
	numattr.setReadable(true);
	numattr.setConnectable(true);
	addAttribute(arestLength1);
	
	arestLength2 = numattr.create("restLength2", "rsl2", MFnNumericData::kDouble, 16.0);
	numattr.setMin(1.0);
	numattr.setKeyable(true); 
	numattr.setReadable(true);
	numattr.setConnectable(true);
	addAttribute(arestLength2);
	
	asoftDistance = numattr.create("softDistance", "sftd", MFnNumericData::kDouble, 1.0);
	numattr.setMin(0.01);
	numattr.setKeyable(true); 
	numattr.setReadable(true);
	numattr.setConnectable(true);
	numattr.setStorable(true);
	addAttribute(asoftDistance);
	
	amaxStretching = numattr.create("maxStretching", "mstc", MFnNumericData::kDouble, 4.0);
	numattr.setMin(0.0);
	numattr.setKeyable(true); 
	numattr.setReadable(true);
	numattr.setConnectable(true);
	numattr.setStorable(true);
	addAttribute(amaxStretching);
	
	attributeAffects(asoftDistance, ik2Bsolver::message);
	
	return MS::kSuccess;
}

MString ik2Bsolver::solverTypeName() const
//
// This method returns the type name used to identify this solver.
//
{
        return MString(kSolverType);
}

MStatus ik2Bsolver::doSolve()
//
// This is the doSolve method which calls solveIK.
//
{
        MStatus stat;

        // Handle Group
        //
        MIkHandleGroup * handle_group = handleGroup();
        if (NULL == handle_group) {
                return MS::kFailure;
        }

        // Handle
        //
        // For single chain types of solvers, get the 0th handle.
        // Single chain solvers are solvers which act on one handle only, 
        // i.e. the     handle group for a single chain solver
        // has only one handle
        //
        MObject handle = handle_group->handle(0);
        MDagPath handlePath = MDagPath::getAPathTo(handle);
        MFnIkHandle handleFn(handlePath, &stat);

        // Effector
        //
        MDagPath effectorPath;
        handleFn.getEffector(effectorPath);
        // MGlobal::displayInfo(effectorPath.fullPathName());
        MFnIkEffector effectorFn(effectorPath);

        // Mid Joint
        //
        effectorPath.pop();
        MFnIkJoint midJointFn(effectorPath);
        
        // End Joint
        //
        MDagPath endJointPath;
        bool hasEndJ = findFirstJointChild(effectorPath, endJointPath);
        // if(hasEndJ) MGlobal::displayInfo(endJointPath.fullPathName());
        
        MFnIkJoint endJointFn(endJointPath);
        
        // Start Joint
        //
        MDagPath startJointPath;
        handleFn.getStartJoint(startJointPath);
        MFnIkJoint startJointFn(startJointPath);

        // Preferred angles
        //
        double startJointPrefAngle[3];
        double midJointPrefAngle[3];
        startJointFn.getPreferedAngle(startJointPrefAngle);
        midJointFn.getPreferedAngle(midJointPrefAngle);

        // Set to preferred angles
        //
        startJointFn.setRotation(startJointPrefAngle, 
                                                         startJointFn.rotationOrder());
        midJointFn.setRotation(midJointPrefAngle, 
                                                   midJointFn.rotationOrder());

        MPoint handlePos = handleFn.rotatePivot(MSpace::kWorld);
        MPoint effectorPos = effectorFn.rotatePivot(MSpace::kWorld);
        MPoint midJointPos = midJointFn.rotatePivot(MSpace::kWorld);
        MPoint startJointPos = startJointFn.rotatePivot(MSpace::kWorld);
        MVector poleVector = poleVectorFromHandle(handlePath);
        poleVector *= handlePath.exclusiveMatrix();
        double twistValue = twistFromHandle(handlePath);
        
		MObject thisNode = thisMObject();
        
        // get rest length
        //
        double restL1, restL2;
		MPlug(thisNode, arestLength1).getValue(restL1);
		MPlug(thisNode, arestLength2).getValue(restL2);
        // get soft distance
        //
        MPlug plug( thisNode, asoftDistance );
        double softD = 0.0;
        plug.getValue(softD);
		
		// get max stretching
		double maxStretching = 0.0;
		MPlug(thisNode, amaxStretching).getValue(maxStretching);
        
        MQuaternion qStart, qMid;
        double stretching = 0.0;
        solveIK(startJointPos,
                        midJointPos,
                        effectorPos,
                        handlePos,
                        poleVector,
                        twistValue,
                        qStart,
                        qMid,
                        softD,
						restL1, restL2,
                        stretching);

        midJointFn.rotateBy(qMid, MSpace::kWorld);
        startJointFn.rotateBy(qStart, MSpace::kWorld);
        
        
		midJointFn.setTranslation(MVector(restL1, 0.0, 0.0), MSpace::kTransform);
		endJointFn.setTranslation(MVector(restL2, 0.0, 0.0), MSpace::kTransform);
			
		if(stretching > maxStretching) stretching = maxStretching;
		if(stretching != 0.0) {
			MVector vstretch(stretching* 0.5, 0.0, 0.0);
			midJointFn.translateBy(vstretch, MSpace::kTransform);
			endJointFn.translateBy(vstretch, MSpace::kTransform);
		}
        
        return MS::kSuccess;
}

void ik2Bsolver::solveIK(const MPoint &startJointPos,
                         const MPoint &midJointPos,
                         const MPoint &effectorPos,
                         const MPoint &handlePos,
                         const MVector &poleVector,
                         double twistValue,
                         MQuaternion &qStart,
                         MQuaternion &qMid,
                         const double & softDistance,
						 const double & restLength1,
						 const double & restLength2,
                         double & stretching)
//
// This is method that actually computes the IK solution.
//
{
        // vector from startJoint to midJoint
        MVector vector1 = midJointPos - startJointPos; vector1 = vector1.normal() * restLength1;
        // vector from midJoint to effector
        MVector vector2 = effectorPos - midJointPos; vector2 = vector2.normal() * restLength2;
        // vector from startJoint to handle
        MVector vectorH = handlePos - startJointPos;
        // vector from startJoint to effector
        MVector vectorE = effectorPos - startJointPos;
        // lengths of those vectors
        const double length1 = restLength1;
        const double length2 = restLength2;
        const double lengthH = vectorH.length();
        // component of the vector1 orthogonal to the vectorE
        const MVector vectorO =
                vector1 - vectorE*((vector1*vectorE)/(vectorE*vectorE));
                
		
        //////////////////////////////////////////////////////////////////
        // calculate q12 which solves for the midJoint rotation
        //////////////////////////////////////////////////////////////////
        // angle between vector1 and vector2
        const double vectorAngle12 = vector1.angle(vector2);
        
        // vector orthogonal to vector1 and 2
        const MVector vectorCross12 = vector1^vector2;
			
        const double lengthHsquared = lengthH * lengthH;
		double weight = 0.0;
		double slowLH = lengthH;// / (1.0 + (6.8 - softDistance) / (restLength1 + restLength2));
		const double da = restLength1 + restLength2 - softDistance;

		if(slowLH > da) {
		    float s = (slowLH - da) / softDistance;
		    if(s> 1.f) s= 1.f;
		    Vector3F pt = m_herm.interpolate(s);
		    MGlobal::displayInfo(MString("herm ")+ pt.y);
			
// approading l1+l2 slower 
//
			weight = 1.0 - exp(-(slowLH - da) / softDistance * 6.98);
			
			weight = pt.y * weight + (1.f - pt.y) * s;
			slowLH = da + softDistance * weight;

			MGlobal::displayInfo(MString("wei ")+weight);
		}

//
//           1
//        /    \
//  l1  /        \ l2
//    /            \	     ---l1---1 ---l2---
//  0 ------ l ------ 2     0 ------ l ------- 2
//
	
// angle for arm extension		
        
		double cos_theta = (slowLH * slowLH - length1*length1 - length2*length2) / (2*length1*length2);
		
		if (cos_theta > 1) 
                cos_theta = 1;
        else if (cos_theta < -1) 
                cos_theta = -1;
		
        const double theta = acos(cos_theta);

        // quaternion for arm extension
        MQuaternion q12(theta - vectorAngle12, vectorCross12);
        
        //////////////////////////////////////////////////////////////////
        // calculate qEH which solves for effector rotating onto the handle
        //////////////////////////////////////////////////////////////////
        // vector2 with quaternion q12 applied
        vector2 = vector2.rotateBy(q12);
        // vectorE with quaternion q12 applied
        vectorE = vector1 + vector2;
        // quaternion for rotating the effector onto the handle
        MQuaternion qEH(vectorE, vectorH);
		
		if(lengthH > vectorE.length()) {
			// MGlobal::displayInfo(MString("vle ")+(lengthH-vectorE.length()));
			stretching = (lengthH-vectorE.length()) * weight;
		}
        
        //////////////////////////////////////////////////////////////////
        // calculate qNP which solves for the rotate plane
        //////////////////////////////////////////////////////////////////
        // vector1 with quaternion qEH applied
        vector1 = vector1.rotateBy(qEH);
        if (vector1.isParallel(vectorH))
                // singular case, use orthogonal component instead
                vector1 = vectorO.rotateBy(qEH);
        // quaternion for rotate plane
        MQuaternion qNP;
        if (!poleVector.isParallel(vectorH) && (lengthHsquared != 0)) {
                // component of vector1 orthogonal to vectorH
                MVector vectorN = 
                        vector1 - vectorH*((vector1*vectorH)/lengthHsquared);
                // component of pole vector orthogonal to vectorH
                MVector vectorP = 
                        poleVector - vectorH*((poleVector*vectorH)/lengthHsquared);
                double dotNP = (vectorN*vectorP)/(vectorN.length()*vectorP.length());
                if (absoluteValue(dotNP + 1.0) < kEpsilon) {
                        // singular case, rotate halfway around vectorH
                        MQuaternion qNP1(kPi, vectorH);
                        qNP = qNP1;
                }
                else {
                        MQuaternion qNP2(vectorN, vectorP);
                        qNP = qNP2;
                }
        }

        //////////////////////////////////////////////////////////////////
        // calculate qTwist which adds the twist
        //////////////////////////////////////////////////////////////////
        MQuaternion qTwist(twistValue, vectorH);
		
		// quaternion for the mid joint
        qMid = q12;     
        // concatenate the quaternions for the start joint
        qStart = qEH*qNP*qTwist;
}

MVector ik2Bsolver::poleVectorFromHandle(const MDagPath &handlePath)
//
// This method returns the pole vector of the IK handle.
//
{
        MStatus stat;
        MFnIkHandle handleFn(handlePath, &stat);
        MPlug pvxPlug = handleFn.findPlug("pvx");
        MPlug pvyPlug = handleFn.findPlug("pvy");
        MPlug pvzPlug = handleFn.findPlug("pvz");
        double pvxValue, pvyValue, pvzValue;
        pvxPlug.getValue(pvxValue);
        pvyPlug.getValue(pvyValue);
        pvzPlug.getValue(pvzValue);
        MVector poleVector(pvxValue, pvyValue, pvzValue);
        return poleVector;
}

double ik2Bsolver::twistFromHandle(const MDagPath &handlePath)
//
// This method returns the twist of the IK handle.
//
{
        MStatus stat;
        MFnIkHandle handleFn(handlePath, &stat);
        MPlug twistPlug = handleFn.findPlug("twist");
        double twistValue;
        twistPlug.getValue(twistValue);
        return twistValue;
}

bool ik2Bsolver::findFirstJointChild(const MDagPath & root, MDagPath & result)
{
    const unsigned count = root.childCount();
    unsigned i;
    for(i=0; i<count; i++) {
        MObject c = root.child(i);
        if(c.hasFn(MFn::kJoint)) {
            MDagPath::getAPathTo(c, result);
            return 1;
        }
    }
    return 0;
}


