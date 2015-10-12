/*
 *  HesperisIO.cpp
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisIO.h"
#include <maya/MGlobal.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnMesh.h>
#include <maya/MPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MFnTransform.h>
#include <maya/MItDag.h>
#include <CurveGroup.h>
#include <ATriangleMeshGroup.h>
#include "HesperisFile.h"
#include <HBase.h>
#include <HWorld.h>
#include <HCurveGroup.h>
#include <HTriangleMeshGroup.h>
#include <AHelper.h>
#include <CurveGroup.h>
#include <sstream>
#include <boost/format.hpp>

bool HesperisIO::WriteTransforms(const MDagPathArray & paths, HesperisFile * file, const std::string & beheadName)
{
    file->clearTransforms();
    
	unsigned i = 0;
	for(;i<paths.length();i++) AddTransform(paths[i], file, beheadName);
	
	file->setDirty();
	file->setWriteComponent(HesperisFile::WTransform);
    bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save transform to file ")+ file->fileName().c_str());
	file->close();
	return true;
}

bool HesperisIO::AddTransform(const MDagPath & path, HesperisFile * file, const std::string & beheadName)
{
	MFnDagNode fdg(path);
	if(fdg.parentCount() < 1) return false;
	
// AHelper::Info<MString>("hes io add transform ", path.fullPathName());

	MObject oparent = fdg.parent(0);
	MFnDagNode fp(oparent);
	MDagPath pp;
	fp.getPath(pp);
	AddTransform(pp, file, beheadName);
	
	std::string nodeName = path.fullPathName().asChar();
	if(beheadName.size() > 1) SHelper::behead(nodeName, beheadName);
    SHelper::removeAnyNamespace(nodeName);
    
// todo extract tm    
	file->addTransform(nodeName, new BaseTransform);
	return true;
}

bool HesperisIO::IsCurveValid(const MDagPath & path)
{
	MStatus stat;
	MFnNurbsCurve fcurve(path.node(), &stat);
	if(!stat) {
		// MGlobal::displayInfo(path.fullPathName() + " is not a curve!");
		return false;
	}
	if(fcurve.numCVs() < 4) {
		MGlobal::displayInfo(path.fullPathName() + " has less than 4 cvs!");
		return false;
	}
	return true;
}

bool HesperisIO::WriteCurves(const MDagPathArray & paths, 
							HesperisFile * file, 
							const std::string & parentName) 
{
    CurveGroup * gcurve = new CurveGroup;
    if(!CreateCurveGroup(paths, gcurve)) {
        MGlobal::displayInfo(" hesperis check curves error");
        return false;
    }
    
	std::string curveName = "|curves";
    if(parentName.size()>1) curveName = boost::str(boost::format("%1%|curves") % parentName);
    SHelper::removeAnyNamespace(curveName);
	
	MGlobal::displayInfo(MString("hes io write curve group ")+curveName.c_str());
    file->clearCurves();
    file->addCurve(curveName, gcurve);
	
	file->setDirty();
	file->setWriteComponent(HesperisFile::WCurve);
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save curves to file ")+ file->fileName().c_str());
	file->close();
	
	return true;
}

bool HesperisIO::WriteMeshes(const MDagPathArray & paths, 
							HesperisFile * file, 
							const std::string & parentName)
{
    ATriangleMeshGroup * combined = new ATriangleMeshGroup;
    if(!CreateMeshGroup(paths, combined)) {
        MGlobal::displayInfo(" hesperis check meshes error");
        return false;
    }
    
    // combined->setDagName(parentName);
    std::string meshName = "|meshes";
    if(parentName.size()>1) meshName = boost::str(boost::format("%1%|meshes") % parentName);
    SHelper::removeAnyNamespace(meshName);

    MGlobal::displayInfo(MString("hes io write mesh group ")+meshName.c_str());
    
    file->clearTriangleMeshes();
    file->addTriangleMesh(meshName, combined);

	file->setDirty();
	file->setWriteComponent(HesperisFile::WTri);
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save mesh to file ")+ file->fileName().c_str());
	file->close();
	
	return true;
}

MMatrix HesperisIO::GetParentTransform(const MDagPath & path)
{
    MMatrix m;
    MDagPath parentPath = path;
    parentPath.pop();
    MStatus stat;
    MFnTransform ft(parentPath, &stat);
    if(!stat) {
        MGlobal::displayWarning(MString("hesperis io cannot create transform func by paht ")+path.fullPathName());
        return m;   
    }
    m = ft.transformation().asMatrix();
    return m;
}

MMatrix HesperisIO::GetWorldTransform(const MDagPath & path)
{
    MMatrix m;
    MDagPath parentPath = path;
    MStatus stat;
    for(;;) {
        stat = parentPath.pop();
        if(!stat) break;
        MFnTransform ft(parentPath, &stat);
        if(!stat) {
            return m;   
        }
        m *= ft.transformation().asMatrix();
    }
    return m;
}

bool HesperisIO::FindNamedChild(MObject & dst, const std::string & name, MObject & oparent)
{
    if(oparent == MObject::kNullObj) {
        MItDag itdag(MItDag::kBreadthFirst);
        oparent = itdag.currentItem();
        MGlobal::displayInfo(MFnDagNode(oparent).name() + " as default parent");
    }
    
    MFnDagNode ppf(oparent);
    for(unsigned i = 0; i <ppf.childCount(); i++) {
        MFnDagNode pf(ppf.child(i));
        std::string curname = pf.name().asChar();
        if(SHelper::isMatched(curname, name)) {
            dst = ppf.child(i);
            return true;
        }
    }
    return false;
}

bool HesperisIO::CreateCurveGroup(const MDagPathArray & paths, CurveGroup * dst)
{
    MStatus stat;
	unsigned i, j;
	int numCvs = 0;
	unsigned numNodes = 0;
    
	const unsigned n = paths.length();
    for(i=0; i<n; i++) {
		if(!IsCurveValid(paths[i])) continue;
		MFnNurbsCurve fcurve(paths[i].node());
		numCvs += fcurve.numCVs();
		numNodes++;
	}
    
    if(numCvs < 4) {
		MGlobal::displayInfo(" too fews cvs!");
		return false;
	}
    
    dst->create(numNodes, numCvs);
    Vector3F * pnts = dst->points();
	unsigned * counts = dst->counts();
    
    unsigned inode = 0;
	unsigned icv = 0;
	unsigned nj;
	MPoint wp;
	MMatrix worldTm;
	
	for(i=0; i<n; i++) {
		if(!IsCurveValid(paths[i])) continue;
		
		worldTm = GetWorldTransform(paths[i]);
		
		MFnNurbsCurve fcurve(paths[i].node());
		nj = fcurve.numCVs();
		MPointArray ps;
		fcurve.getCVs(ps, MSpace::kWorld);
		
		counts[inode] = nj;
		inode++;
		
		for(j=0; j<nj; j++) {
			wp = ps[j] * worldTm;
			pnts[icv].set((float)wp.x, (float)wp.y, (float)wp.z);
			icv++;
		}
	}
    return true;
}

bool HesperisIO::CreateMeshGroup(const MDagPathArray & paths, ATriangleMeshGroup * dst)
{
    MStatus stat;
	unsigned i, j;
	int numPnts = 0;
	unsigned numNodes = 0;
	unsigned numTris = 0;
	
	// MGlobal::displayInfo(" hesperis check meshes");
	
	MIntArray triangleCounts, triangleVertices;
	MPointArray ps;
    MPoint wp;
	MMatrix worldTm;
    
    const unsigned n = paths.length();
    for(i=0; i<n; i++) {
		MFnMesh fmesh(paths[i].node(), &stat);
		if(!stat) continue;
		numNodes++;
        
        numPnts += fmesh.numVertices();
		fmesh.getTriangles(triangleCounts, triangleVertices);
		numTris += triangleVertices.length() / 3;
	}
    
    if(numNodes < 1 || numTris < 1) {
        MGlobal::displayInfo(" insufficient mesh data");
        return false;   
    }
    
    //MGlobal::displayInfo(MString(" mesh count: ") + numNodes +
    //                     MString(" vertex count: ") + numPnts +
	//                    MString(" triangle count: ") + numTris);
	
    dst->create(numPnts, numTris, numNodes);
	Vector3F * pnts = dst->points();
	unsigned * inds = dst->indices();
    unsigned * pntDrift = dst->pointDrifts();
    unsigned * indDrift = dst->indexDrifts();
    
    unsigned pDrift = 0;
    unsigned iDrift = 0;
    unsigned iNode = 0;
    for(i=0; i<n; i++) {
		MFnMesh fmesh(paths[i].node(), &stat);
		if(!stat) continue;
        //MGlobal::displayInfo(fmesh.name());
        //MGlobal::displayInfo(MString("p drift ")+pDrift+
        //                     MString("i drift ")+iDrift);
		
        worldTm = GetWorldTransform(paths[i]);
		
		fmesh.getPoints(ps, MSpace::kObject);
		fmesh.getTriangles(triangleCounts, triangleVertices);
			
		for(j=0; j<fmesh.numVertices(); j++) {
            wp = ps[j] * worldTm;
			pnts[pDrift + j].set((float)wp.x, (float)wp.y, (float)wp.z);
        }
		
		for(j=0; j<triangleVertices.length(); j++)
			inds[iDrift + j] = pDrift + triangleVertices[j];
        
        pntDrift[iNode] = pDrift;
        indDrift[iNode] = iDrift;
        
        pDrift += fmesh.numVertices();
        iDrift += triangleVertices.length();
        iNode++;
	}
    
    return true;
}

bool HesperisIO::LsCurves(std::vector<std::string > & dst)
{
    HWorld grpWorld;
    HesperisFile::LsNames<HCurveGroup>(dst, &grpWorld);
    grpWorld.close();
    return true;   
}

bool HesperisIO::LsMeshes(std::vector<std::string > & dst)
{
    HWorld grpWorld;
    HesperisFile::LsNames<HTriangleMeshGroup>(dst, &grpWorld);
    grpWorld.close();
    return true;
}

bool HesperisIO::LsTransforms(std::vector<std::string > & dst)
{
    HWorld grpWorld;
    HesperisFile::LsNames<HTransform>(dst, &grpWorld);
    grpWorld.close();
    return true;
}

bool HesperisIO::GetTransform(BaseTransform * dst, const MDagPath & path)
{
	MStatus stat;
	MFnTransform ftransform(path, &stat);
	if(!stat) {
		MGlobal::displayInfo(MString("is not transform ")+path.fullPathName());
		return false;
	}
	
	MPoint mRotatePivot, mScalePivot;
	MVector mTranslate, mRotatePivotTranslate, mScalePivotTranslate;
    double mRotationInRadians[3];
    double mScales[3];
	MTransformationMatrix::RotationOrder mRotationOrder;
    
	mTranslate = ftransform.getTranslation(MSpace::kTransform);
    mScalePivot = ftransform.scalePivot(MSpace::kTransform);
    mRotatePivot = ftransform.rotatePivot(MSpace::kTransform);
    mRotatePivotTranslate = ftransform.rotatePivotTranslation(MSpace::kTransform);
    mScalePivotTranslate = ftransform.scalePivotTranslation(MSpace::kTransform);
    mRotationOrder = ftransform.rotationOrder();
    ftransform.getRotation(mRotationInRadians, mRotationOrder);
    ftransform.getScale(mScales);
	
    //AHelper::PrintMatrix("matrix", ftransform.transformation().asMatrix());

	dst->setTranslation(Vector3F(mTranslate.x, mTranslate.y, mTranslate.z));
	dst->setRotationAngles(Vector3F(mRotationInRadians[0], mRotationInRadians[1], mRotationInRadians[2]));
	dst->setScale(Vector3F(mScales[0], mScales[1], mScales[2]));
	dst->setRotationOrder(GetRotationOrder(mRotationOrder));
	dst->setRotatePivot(Vector3F(mRotatePivot.x, mRotatePivot.y, mRotatePivot.z), Vector3F(mRotatePivotTranslate.x, mRotatePivotTranslate.y, mRotatePivotTranslate.z));
	dst->setScalePivot(Vector3F(mScalePivot.x, mScalePivot.y, mScalePivot.z), Vector3F(mScalePivotTranslate.x, mScalePivotTranslate.y, mScalePivotTranslate.z));
	
	//AHelper::Info<Matrix44F>("space", dst->space());
	
	return true;
}

Matrix33F::RotateOrder HesperisIO::GetRotationOrder(MTransformationMatrix::RotationOrder x)
{
	Matrix33F::RotateOrder r = Matrix33F::Unknown;
	switch (x) {
		case MTransformationMatrix::kXYZ:
			r = Matrix33F::XYZ;
			break;
		case MTransformationMatrix::kYZX:
			r = Matrix33F::YZX;
			break;
		case MTransformationMatrix::kZXY:
			r = Matrix33F::ZXY;
			break;
		case MTransformationMatrix::kXZY:
			r = Matrix33F::XZY;
			break;
		case MTransformationMatrix::kYXZ:
			r = Matrix33F::YXZ;
			break;
		case MTransformationMatrix::kZYX:
			r = Matrix33F::ZYX;
			break;
		default:
			break;
	}
	return r;
}

MObject HesperisTransformCreator::create(BaseTransform * data, MObject & parentObj,
                       const std::string & nodeName)
{
    MObject otm = MObject::kNullObj;
    if(!HesperisIO::FindNamedChild(otm, nodeName, parentObj)) {
        MFnTransform ftransform;
        otm = ftransform.create(parentObj);

        std::string validName(nodeName);
        SHelper::noColon(validName);
        ftransform.setName(validName.c_str()); 
    }
    // MGlobal::displayInfo(MString("todo transform in ")+nodeName.c_str()); 
    return otm;
}
//:~