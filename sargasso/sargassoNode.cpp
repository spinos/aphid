#include "sargassoNode.h"
#include <maya/MFnMatrixData.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnMeshData.h>
#include <maya/MFnMesh.h>
#include <AHelper.h>
#include <ATriangleMesh.h>
#include <TriangleDifference.h>
#include <BaseBuffer.h>
MTypeId     SargassoNode::id( 0x9b1798 );
MObject     SargassoNode::compoundOutput;  
MObject		SargassoNode::targetTransform;      
MObject     SargassoNode::targetWeight;    
MObject		SargassoNode::targetOffset;
MObject		SargassoNode::targetRestP;
MObject     SargassoNode::aconstraintParentInverseMatrix;       
MObject		SargassoNode::constraintTranslateX;
MObject		SargassoNode::constraintTranslateY;
MObject		SargassoNode::constraintTranslateZ;
MObject SargassoNode::atargetRestP;
MObject SargassoNode::atargetTri;
MObject SargassoNode::atargetNv;
MObject SargassoNode::atargetNt;
MObject SargassoNode::aobjN;
MObject SargassoNode::aobjLocal;
MObject SargassoNode::atargetBind;
MObject SargassoNode::atargetMesh;
	
SargassoNode::SargassoNode() 
{
    m_isInitd = false;
    m_mesh = new ATriangleMesh;
    m_diff = 0;
    m_localP = new BaseBuffer;
}

SargassoNode::~SargassoNode() 
{
    delete m_mesh;
    if(m_diff) delete m_diff;
    delete m_localP;
}

void SargassoNode::postConstructor()
{
}

MStatus SargassoNode::compute( const MPlug& plug, MDataBlock& block )
{	
	MStatus returnStatus;
/* 
    if(plug == constraintTranslateX || plug == constraintTranslateY || plug == constraintTranslateZ) {
        if(!m_isInitd) {
// read rest position
            MDataHandle htgo = block.inputValue(targetRestP);
            double3 & tgo = htgo.asDouble3();
            MGlobal::displayInfo(MString("target rest p ")+tgo[0]+" "+tgo[1]+" "+tgo[2]);
            m_restPos = MPoint(tgo[0],tgo[1],tgo[2]);
			m_isInitd = true;
		}
		
		MArrayDataHandle targetArray = block.inputArrayValue( compoundOutput );
		const unsigned int targetArrayCount = targetArray.elementCount();
        MMatrix tm;
        tm.setToIdentity();
        unsigned int i;
		for ( i = 0; i < targetArrayCount; i++ ) {
            MDataHandle targetElement = targetArray.inputValue(&returnStatus);
            if(!returnStatus) {
                MGlobal::displayInfo("failed to get input value target element");
            }
            MDataHandle htm = targetElement.child(targetTransform);
            MFnMatrixData ftm(htm.data(), &returnStatus);
            if(!returnStatus) {
                MGlobal::displayInfo("failed to get matrix data");
            }
            tm = ftm.matrix();
            targetArray.next();
        }
		
		MDataHandle hparentInvMat = block.inputValue(aconstraintParentInverseMatrix);
		MMatrix parentInvMat = hparentInvMat.asMatrix();

// world position
        MPoint curPos(tm(3,0), tm(3,1), tm(3,2));
// offset in local space
		m_offsetToRest = m_restPos - curPos;
// object position in world space
		MPoint localP = m_offsetToRest * tm + curPos;
// in local space
		localP *= parentInvMat;

        MDataHandle hout;
        if(plug == constraintTranslateX) {
            hout = block.outputValue(constraintTranslateX);
			hout.set(localP.x);
        }
        else if(plug == constraintTranslateY) {
            hout = block.outputValue(constraintTranslateY);
			hout.set(localP.y);
        }
        else if(plug == constraintTranslateZ) {
            hout = block.outputValue(constraintTranslateZ);
			hout.set(localP.z);
        }
		
		//MPlug pgTx(thisMObject(), constraintTargetX);
		//pgTx.setValue(m_lastPos.x);
		//MPlug pgTy(thisMObject(), constraintTargetY);
		//pgTy.setValue(m_lastPos.y);
		//MPlug pgTz(thisMObject(), constraintTargetZ);
		//pgTz.setValue(m_lastPos.z);
		
		MPlug pgOx(thisMObject(), constraintObjectX);
		pgOx.setValue(m_offsetToRest.x);
		MPlug pgOy(thisMObject(), constraintObjectY);
		pgOy.setValue(m_offsetToRest.y);
		MPlug pgOz(thisMObject(), constraintObjectZ);
		pgOz.setValue(m_offsetToRest.z);
        
       // MFnNumericData nd;
		//MObject offsetData = nd.create( MFnNumericData::k3Double);
        //nd.setData3Double(m_lastPos.x, m_lastPos.y, m_lastPos.z);
        //MPlug pgTgo(thisMObject(), targetOffset);
        //pgTgo.setValue(offsetData); 
    }
	else
		return MS::kUnknownParameter;
*/
	return MS::kSuccess;
}

void* SargassoNode::creator()
{
	return new SargassoNode;
}

MStatus SargassoNode::initialize()
{
	MFnNumericAttribute nAttr;
	MStatus				status;

	// constraint attributes

// Parent inverse matrix: delete on disconnect
	MFnTypedAttribute typedAttr;
    
    MFnTypedAttribute pimAttr;
    aconstraintParentInverseMatrix = pimAttr.create( "constraintParentInvMat", "cpim", MFnData::kMatrix, &status );
    pimAttr.setArray(true);	
    pimAttr.setStorable(false);	
    pimAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	
    status = addAttribute(aconstraintParentInverseMatrix);
	if (!status) { status.perror("addAttribute"); return status;}
    
    targetTransform = typedAttr.create( "targetTransform", "ttm", MFnData::kMatrix, &status ); 	
    if (!status) { status.perror("typedAttr.create:targetTransform"); return status;}
    status = typedAttr.setDisconnectBehavior(MFnAttribute::kDelete);
    if (!status) { status.perror("typedAttr.setDisconnectBehavior:targetTransform"); return status;}
    
// Target weight: double, min 0, default 1.0, keyable, delete on disconnect
	MFnNumericAttribute typedAttrKeyable;
		SargassoNode::targetWeight 
			= typedAttrKeyable.create( "weight", "wt", MFnNumericData::kDouble, 1.0, &status );
		if (!status) { status.perror("typedAttrKeyable.create:weight"); return status;}
		status = typedAttrKeyable.setMin( (double) 0 );
		if (!status) { status.perror("typedAttrKeyable.setMin"); return status;}
		status = typedAttrKeyable.setKeyable( true );
		if (!status) { status.perror("typedAttrKeyable.setKeyable"); return status;}
		status = typedAttrKeyable.setDisconnectBehavior(MFnAttribute::kDelete);
		if (!status) { status.perror("typedAttrKeyable.setDisconnectBehavior:cgeom"); return status;}

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
    
	{	// Compound target(geometry,weight): array, delete on disconnect
		MFnCompoundAttribute compoundAttr;
		compoundOutput = compoundAttr.create( "outValue", "otv",&status );
		if (!status) { status.perror("compoundAttr.create"); return status;}
        status = compoundAttr.addChild( constraintTranslateX );
		if (!status) { status.perror("compoundAttr.addChild tx"); return status;}
		status = compoundAttr.addChild( constraintTranslateY );
		if (!status) { status.perror("compoundAttr.addChild ty"); return status;}
		status = compoundAttr.addChild( constraintTranslateZ );
		if (!status) { status.perror("compoundAttr.addChild tz"); return status;}
		
        compoundAttr.setArray( true );
		//status = compoundAttr.setDisconnectBehavior(MFnAttribute::kDelete);
		//if (!status) { status.perror("typedAttrKeyable.setDisconnectBehavior:cgeom"); return status;}
	}

    targetOffset = numAttr.create("targetOffset", "tgo", MFnNumericData::k3Double, 0.0, &status);
    if (!status) { status.perror("addAttribute targetOffset"); return status;}
    addAttribute(targetOffset);
    
    targetRestP = numAttr.create("targetRestAt", "tgrt", MFnNumericData::k3Double, 0.0, &status);
    if (!status) { status.perror("addAttribute targetRestAt"); return status;}
    addAttribute(targetRestP);

	status = addAttribute( compoundOutput );
	if (!status) { status.perror("addAttribute"); return status;}
    
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	
	atargetRestP = typedAttr.create( "targetRestP", "tgrp", MFnData::kPointArray, pntArrayDataFn.object());
 	typedAttr.setStorable(true);
 	addAttribute(atargetRestP);
	
	MIntArray defaultIntArray;
	MFnIntArrayData intArrayDataFn;
	intArrayDataFn.create( defaultIntArray );
	
	atargetTri = typedAttr.create( "targetTriangle", "tgtri", MFnData::kIntArray, intArrayDataFn.object());
 	typedAttr.setStorable(true);
 	addAttribute(atargetTri);
	
	atargetBind = typedAttr.create( "targetBindId", "tgbdi", MFnData::kIntArray, intArrayDataFn.object());
 	typedAttr.setStorable(true);
 	addAttribute(atargetBind);
	
	atargetNv = numAttr.create( "targetNumV", "tgnv", MFnNumericData::kInt, 0, &status );
    addAttribute(atargetNv);
	
	atargetNt = numAttr.create( "targetNumTri", "tgnt", MFnNumericData::kInt, 0, &status );
    addAttribute(atargetNt);
	
	MVectorArray defaultVectArray;
	MFnVectorArrayData vectArrayDataFn;
	vectArrayDataFn.create( defaultVectArray );
	
	aobjLocal = typedAttr.create( "objectLocalP", "oblp", MFnData::kVectorArray, vectArrayDataFn.object());
 	typedAttr.setStorable(true);
 	addAttribute(aobjLocal);
	
	aobjN = numAttr.create( "objectCount", "obct", MFnNumericData::kInt, 0, &status );
    addAttribute(aobjN);
	
	atargetMesh = typedAttr.create("targetMesh", "tgms", MFnMeshData::kMesh, &status);
	typedAttr.setStorable(false);
	addAttribute(atargetMesh);
	
    //attributeAffects(targetTransform, constraintTranslateX);
    //attributeAffects(targetTransform, constraintTranslateY);
    //attributeAffects(targetTransform, constraintTranslateZ);

	return MS::kSuccess;
}

MStatus SargassoNode::connectionMade(const MPlug &plug, const MPlug &otherPlug, bool asSrc)
{
    if ( plug == atargetMesh ) {
        MStatus stat;
        MObject thisObj = thisMObject();
        MObject val;
        plug.getValue(val);
        creatRestShape(val);
    }

    return MPxNode::connectionMade( plug, otherPlug, asSrc );
}

bool SargassoNode::creatRestShape(const MObject & m)
{
    MStatus stat;
    MFnMesh fmesh(m, &stat);
    if(!stat) {
        MGlobal::displayInfo("val is not mesh");
        return false;
    }
    MObject thisObj = thisMObject();
    MPlug pnv(thisObj, atargetNv);
    const int restNv = pnv.asInt();
    
    if(restNv != fmesh.numVertices()) {
        MGlobal::displayInfo("target nv not match");
        return false;
    }
    
    MPlug pntriv(thisObj, atargetNt);
    const int restNtriv = pntriv.asInt();
    
    MIntArray triangleCounts, triangleVertices;
	fmesh.getTriangles(triangleCounts, triangleVertices);
	
    if(restNtriv != triangleVertices.length()) {
        MGlobal::displayInfo("target ntri not match");
        return false;
    }
    
    m_mesh->create(restNv, restNtriv/3);
    
    MPlug prestp(thisObj, atargetRestP);
    MObject orestp;
    prestp.getValue(orestp);
    MFnPointArrayData frestp(orestp);
    MPointArray restPArray = frestp.array();	
    
    if(restPArray.length() != restNv) {
        MGlobal::displayInfo("cached target nv not match");
        return false;
    }
    
    Vector3F * p = m_mesh->points();
    unsigned i=0;
    for(;i<restNv;i++) p[i].set(restPArray[i].x, restPArray[i].y, restPArray[i].z);
    
    MPlug presttri(thisObj, atargetTri);
    MObject oresttri;
    presttri.getValue(oresttri);
    MFnIntArrayData fresttri(oresttri);
    MIntArray restTriArray = fresttri.array();	
    
    if(restTriArray.length() != restNtriv) {
        MGlobal::displayInfo("cached target ntri not match");
        return false;
    }
    
    unsigned * ind = m_mesh->indices();
    for(i=0;i<restNtriv;i++) ind[i] = restTriArray[i];
    
    const BoundingBox box = ((AGenericMesh *)m_mesh)->calculateBBox();
    AHelper::Info<BoundingBox>("target mesh box", box);
    
    m_diff = new TriangleDifference(m_mesh);
    
    MPlug ptargetbind(thisObj, atargetBind);
    MObject otargetbind;
    ptargetbind.getValue(otargetbind);
    MFnIntArrayData ftargetbind(otargetbind);
    MIntArray targetBindArray = ftargetbind.array();
    const unsigned nbind = targetBindArray.length();
    AHelper::Info<unsigned>("n binded triangles", nbind);
    
    std::vector<unsigned> vbind;
    for(i=0;i<nbind;i++) vbind.push_back(targetBindArray[i]);
    m_diff->requireQ(vbind);
    vbind.clear();
    
    MPlug pobjLocal(thisObj, aobjLocal);
    MObject oobjLocal;
    pobjLocal.getValue(oobjLocal);
    MFnVectorArrayData fobjLocal(oobjLocal);
    MVectorArray localPArray = fobjLocal.array();
    const unsigned nlocalP = localPArray.length();
    AHelper::Info<unsigned>("n constrained objects", nlocalP);
    
    m_localP->create(nlocalP * 12);
    
    Vector3F * lp = localP();
    for(i=0;i<nlocalP;i++) lp[i].set(localPArray[i].x, localPArray[i].y, localPArray[i].z);
    
    return true;
}

Vector3F * SargassoNode::localP()
{ return (Vector3F *)m_localP->data(); }
//:~
