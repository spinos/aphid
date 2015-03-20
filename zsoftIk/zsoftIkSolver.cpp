#include "zsoftIkSolver.h"
#include <maya/MFnNumericAttribute.h>

MTypeId ik2Bsolver::id(0x8fa7476);

MObject ik2Bsolver::asoftDistance;

ik2Bsolver::ik2Bsolver()
        : MPxIkSolverNode()
{
    m_isFirst = 1;
    m_isExtending = 0;
    m_lastLe = 0.f;
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
    asoftDistance = numattr.create("softDistance", "sftd", MFnNumericData::kDouble, 0.0);
	numattr.setKeyable(true); 
	numattr.setReadable(true);
	numattr.setConnectable(true);
	numattr.setStorable(true);
	addAttribute(asoftDistance);
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
        MGlobal::displayInfo(effectorPath.fullPathName());
        MFnIkEffector effectorFn(effectorPath);

        // Mid Joint
        //
        effectorPath.pop();
        MFnIkJoint midJointFn(effectorPath);
        
        // End Joint
        //
        MDagPath endJointPath;
        bool hasEndJ = findFirstJointChild(effectorPath, endJointPath);
        if(hasEndJ) MGlobal::displayInfo(endJointPath.fullPathName());
        
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
        
        // get rest length
        //
        if(m_isFirst) {
            m_restLength[0] =  (midJointPos - startJointPos).length();
            m_restLength[1] =  (effectorPos - midJointPos).length();
            
            MGlobal::displayInfo(MString("rest length: ")+ m_restLength[0]+ ","+m_restLength[1]);
            m_lastLe =  (handlePos - startJointPos).length();
            m_isFirst = 0;
        }
        
        // get soft distance
        //
        MObject thisNode = thisMObject();
        MPlug plug( thisNode, asoftDistance );
        double softD = 0.0;
        plug.getValue(softD);
        MGlobal::displayInfo(MString("sd")+softD);
        
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
                        stretching);

        midJointFn.rotateBy(qMid, MSpace::kWorld);
        startJointFn.rotateBy(qStart, MSpace::kWorld);
        
        /*
		if(stretching != 0.0) {
			MVector vstretch(stretching* 0.5, 0.0, 0.0);
			midJointFn.translateBy(vstretch, MSpace::kObject);
			endJointFn.translateBy(vstretch, MSpace::kObject);
		}
		*/
        
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
                         double & stretching)
//
// This is method that actually computes the IK solution.
//
{
        // vector from startJoint to midJoint
        MVector vector1 = midJointPos - startJointPos;
        // vector from midJoint to effector
        MVector vector2 = effectorPos - midJointPos;
        // vector from startJoint to handle
        MVector vectorH = handlePos - startJointPos;
        // vector from startJoint to effector
        MVector vectorE = effectorPos - startJointPos;
        // lengths of those vectors
        double length1 = vector1.length();
        double length2 = vector2.length();
        double lengthH = vectorH.length();
        // component of the vector1 orthogonal to the vectorE
        MVector vectorO =
                vector1 - vectorE*((vector1*vectorE)/(vectorE*vectorE));
                
        double div = vectorH.length() / vectorE.length();
        //MGlobal::displayInfo(MString("div ")+div);
        /*MVector midJoint2Effecor = effectorPos - midJointPos;
        MGlobal::displayInfo(MString("effector to mid length ")+midJoint2Effecor.length());
        MGlobal::displayInfo(MString("start to mid length ")+ vector1.length());
        MGlobal::displayInfo(MString("start to effector length ")+ vectorE.length());
        
        MGlobal::displayInfo(MString("start to handle length ")+ vectorH.length());*/
        
        MGlobal::displayInfo(MString("soft + le / rest ")+ vectorH.length() + " + " + softDistance + " / " + (m_restLength[0] + m_restLength[1]) + " = " + (vectorH.length() + softDistance) / (m_restLength[0] + m_restLength[1]) );

        m_isExtending = (lengthH >= m_lastLe);
        if(m_isExtending) MGlobal::displayInfo("extending");
        else MGlobal::displayInfo("compressing");
        
        double dL = lengthH - m_lastLe;
        m_lastLe = lengthH;
        
        double factor = (vectorH.length()  + softDistance - (m_restLength[0] + m_restLength[1]) ) / softDistance * 0.5;
        if(m_isExtending) {
            factor = 1.0 - factor;
            if(factor < 0.0) factor = 0.0;
            if(factor > 1.0) factor = 0.0;  
        }
        else {
            if(factor < 0.0) factor = 0.0;
            if(factor > 1.0) factor = 0.0;
        }

        if(((vectorH.length() + softDistance) / (m_restLength[0] + m_restLength[1])) > 1.0) {
            MGlobal::displayInfo(MString("stretch length ")+(vectorH.length() + softDistance - (m_restLength[0] + m_restLength[1])) );
            MGlobal::displayInfo(MString("stretch factor ")+factor);
            stretching = dL * factor;
            length1 += stretching * 0.5;
            length2 += stretching * 0.5;
            // length1 = m_restLength[0] + stretching * 0.5;
            // length2 = m_restLength[1] + stretching * 0.5;
            //
            vector1 = vector1.normal() * length1;
            vector2 = vector2.normal() * length2;
        }
        
        //////////////////////////////////////////////////////////////////
        // calculate q12 which solves for the midJoint rotation
        //////////////////////////////////////////////////////////////////
        // angle between vector1 and vector2
        double vectorAngle12 = vector1.angle(vector2);
        
        // vector orthogonal to vector1 and 2
        MVector vectorCross12 = vector1^vector2;
        double lengthHsquared = lengthH*lengthH;
        // angle for arm extension 
        double cos_theta = 
                (lengthHsquared - length1*length1 - length2*length2)
                /(2*length1*length2);
        if (cos_theta > 1) 
                cos_theta = 1;
        else if (cos_theta < -1) 
                cos_theta = -1;
        double theta = acos(cos_theta);
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
        
        MGlobal::displayInfo(MString("angle ")+vector1.angle(vector2));

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

//////////////////////////////////////////////////////////////////
//
// IK 2 Bone Solver Callbacks
//
//////////////////////////////////////////////////////////////////

MCallbackId addIK2BsolverCallbacks::afterNewId;
MCallbackId addIK2BsolverCallbacks::afterOpenId;                                                                                                                        

void *addIK2BsolverCallbacks::creator()
{
        return new addIK2BsolverCallbacks;
}

void createIK2BsolverAfterNew(void *clientData)
//
// This method creates the ik2Bsolver after a File->New.
//
{
        MSelectionList selList;
        MGlobal::getActiveSelectionList( selList );
        MGlobal::executeCommand("createNode -n zik2Bsolver zik2Bsolver");
        MGlobal::setActiveSelectionList( selList );
}

void createIK2BsolverAfterOpen(void *clientData)
//
// This method creates the ik2Bsolver after a File->Open
// if the ik2Bsolver does not exist in the loaded file.
//
{
        MSelectionList selList;
        MGlobal::getSelectionListByName("zik2Bsolver", selList);
        if (selList.length() == 0) {
                MGlobal::getActiveSelectionList( selList );
                MGlobal::executeCommand("createNode -n zik2Bsolver zik2Bsolver");
                MGlobal::setActiveSelectionList( selList );
        }
}

MStatus addIK2BsolverCallbacks::doIt(const MArgList &args)
//
// This method adds the File->New and File->Open callbacks
// used to recreate the ik2Bsolver.
//
{
    // Get the callback IDs so we can deregister them 
        // when the plug-in is unloaded.
        afterNewId = MSceneMessage::addCallback(MSceneMessage::kAfterNew, 
                                                           createIK2BsolverAfterNew);
        afterOpenId = MSceneMessage::addCallback(MSceneMessage::kAfterOpen, 
                                                           createIK2BsolverAfterOpen);
        return MS::kSuccess;
}



