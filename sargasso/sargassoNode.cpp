#include "sargassoNode.h"
#include <maya/MFnMatrixData.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnMeshData.h>
#include <maya/MFnMesh.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MTransformationMatrix.h>
#include <AHelper.h>
#include <ATriangleMesh.h>
#include <TriangleDifference.h>
#include <BaseBuffer.h>
MTypeId     SargassoNode::id( 0x9b1798 );
MObject     SargassoNode::compoundOutput;
MObject     SargassoNode::aconstraintParentInverseMatrix;       
MObject		SargassoNode::constraintTranslateX;
MObject		SargassoNode::constraintTranslateY;
MObject		SargassoNode::constraintTranslateZ;
MObject	SargassoNode::constraintRotateX;
MObject	SargassoNode::constraintRotateY;
MObject	SargassoNode::constraintRotateZ;
MObject SargassoNode::atargetRestP;
MObject SargassoNode::atargetTri;
MObject SargassoNode::atargetNv;
MObject SargassoNode::atargetNt;
MObject SargassoNode::aobjN;
MObject SargassoNode::aobjLocal;
MObject SargassoNode::atargetBind;
MObject SargassoNode::atargetMesh;
MObject SargassoNode::aobjTri;

SargassoNode::SargassoNode() 
{
    m_isInitd = false;
    m_mesh = new ATriangleMesh;
    m_diff = 0;
    m_localP = new BaseBuffer;
    m_triId = new BaseBuffer;
    m_numObjects = 0;
}

SargassoNode::~SargassoNode() 
{
    delete m_mesh;
    if(m_diff) delete m_diff;
    delete m_localP;
    delete m_triId;
}

void SargassoNode::postConstructor()
{
}

MStatus SargassoNode::compute( const MPlug& plug, MDataBlock& block )
{	
	MStatus stat;
    
    if(plug == constraintRotateX || 
        plug == constraintRotateY ||
        plug == constraintRotateZ ||
        plug == constraintTranslateX || 
        plug == constraintTranslateY ||
        plug == constraintTranslateZ) {
         
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
         
         unsigned itri = objectTriangleInd()[iobject];
         
         const Vector3F objectP = localP()[iobject];
         
         Matrix33F q = m_diff->Q()[itri];
         q.orthoNormalize();
         const Vector3F t = m_mesh->triangleCenter(itri);
         Matrix44F sp;
         sp.setRotation(q);
		 sp.setTranslation(t);
         Vector3F solvedT = sp.transform(objectP);
         
         MMatrix mat;
         AHelper::ConvertToMMatrix(mat, sp);
         MTransformationMatrix mtm(mat);
         
         double rot[3];
         MTransformationMatrix::RotationOrder rotorder =  MTransformationMatrix::kXYZ;
         mtm.getRotation(rot, rotorder);
         
         MDataHandle hout = block.outputValue(plug, &stat);
         if(!stat) AHelper::Info<MString>("cannot get output value", plug.parent().name());
             
         if(plug == constraintTranslateX) {
             hout.set((double)solvedT.x);
         }
         else if(plug == constraintTranslateY) {
             hout.set((double)solvedT.y);
         }
         else if(plug == constraintTranslateZ) {
             hout.set((double)solvedT.z);
         }
         else if(plug == constraintRotateX) {
             hout.set(rot[0]);
         }
         else if(plug == constraintRotateY) {
             hout.set(rot[1]);
         }
         else if(plug == constraintRotateZ) {
             hout.set(rot[2]);
         }
         block.setClean( plug );
    }
	else
		return MS::kUnknownParameter;

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

	MFnTypedAttribute typedAttr;
    
    MFnTypedAttribute pimAttr;
    aconstraintParentInverseMatrix = pimAttr.create( "constraintParentInvMat", "cpim", MFnData::kMatrix, &status );
    pimAttr.setArray(true);	
    pimAttr.setStorable(false);	
    pimAttr.setDisconnectBehavior(MFnAttribute::kDelete);
	
    status = addAttribute(aconstraintParentInverseMatrix);
	if (!status) { status.perror("addAttribute"); return status;}
    
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
    
    MFnUnitAttribute angleAttr;
    constraintRotateX = angleAttr.create( "constraintRotateX", "crx", MFnUnitAttribute::kAngle , 0.0, &status );
    if(!status) {
        MGlobal::displayInfo("failed to create attrib constraintRotateX");
        return status;
    }
    
    constraintRotateY = angleAttr.create( "constraintRotateY", "cry", MFnUnitAttribute::kAngle , 0.0, &status );
    if(!status) {
        MGlobal::displayInfo("failed to create attrib constraintRotateY");
        return status;
    }
    
    constraintRotateZ = angleAttr.create( "constraintRotateZ", "crz", MFnUnitAttribute::kAngle , 0.0, &status );
    if(!status) {
        MGlobal::displayInfo("failed to create attrib constraintRotateY");
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
		compoundAttr.addChild( constraintRotateX );
        compoundAttr.addChild( constraintRotateY );
        compoundAttr.addChild( constraintRotateZ );
        compoundAttr.setArray( true );
		//status = compoundAttr.setDisconnectBehavior(MFnAttribute::kDelete);
		//if (!status) { status.perror("typedAttrKeyable.setDisconnectBehavior:cgeom"); return status;}
	}

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
    
    aobjTri = typedAttr.create( "objectTriId", "obti", MFnData::kIntArray, intArrayDataFn.object());
 	typedAttr.setStorable(true);
 	addAttribute(aobjTri);
	
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
	
    attributeAffects(atargetMesh, compoundOutput);
    attributeAffects(aconstraintParentInverseMatrix, compoundOutput);

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
    m_numObjects = localPArray.length();
    AHelper::Info<unsigned>("n constrained objects", m_numObjects);
    
    m_localP->create(m_numObjects * 12);
    
    Vector3F * lp = localP();
    for(i=0;i<m_numObjects;i++) lp[i].set(localPArray[i].x, localPArray[i].y, localPArray[i].z);
    
    m_triId->create(m_numObjects * 4);
    
    MPlug pobjtri(thisObj, aobjTri);
    MObject oobjtri;
    pobjtri.getValue(oobjtri);
    MFnIntArrayData fobjtri(oobjtri);
    MIntArray objtriArray = fobjtri.array();
    
    unsigned * tri = objectTriangleInd();
    for(i=0;i<m_numObjects;i++) tri[i] = objtriArray[i];
        
    return true;
}

bool SargassoNode::updateShape(const MObject & m)
{
    MStatus stat;
    MFnMesh fmesh(m, &stat);
    if(!stat) {
        MGlobal::displayInfo("val is not mesh");
        return false;
    }
    
    MPointArray ps;
	fmesh.getPoints(ps, MSpace::kWorld);
	
    const unsigned n = m_mesh->numPoints();
    Vector3F * p = m_mesh->points();
    unsigned i=0;
    for(;i<n;i++) p[i].set(ps[i].x, ps[i].y, ps[i].z);
    
    m_diff->computeQ(m_mesh);
    
   // MGlobal::displayInfo("update mesh");
    return true;
}

Vector3F * SargassoNode::localP()
{ return (Vector3F *)m_localP->data(); }

unsigned * SargassoNode::objectTriangleInd()
{ return (unsigned *)m_triId->data(); }
//:~
