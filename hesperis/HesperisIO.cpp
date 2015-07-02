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
#include <ATriangleMesh.h>
#include "HesperisFile.h"
#include <HBase.h>
#include <HWorld.h>
#include <HTransform.h>
#include <HCurveGroup.h>
#include <SHelper.h>
#include <CurveGroup.h>
#include <BaseTransform.h>
#include <sstream>

bool HesperisIO::WriteTransforms(const MDagPathArray & paths, HesperisFile * file, const std::string & beheadName)
{
	file->setWriteComponent(HesperisFile::WTransform);
    file->setDirty();
	
	unsigned i = 0;
	for(;i<paths.length();i++) AddTransform(paths[i], file, beheadName);
	
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save transform to file ")+ file->fileName().c_str());
	file->close();
	return true;
}

bool HesperisIO::AddTransform(const MDagPath & path, HesperisFile * file, const std::string & beheadName)
{
	MFnDagNode fdg(path);
	if(fdg.parentCount()>0) {
		MObject oparent = fdg.parent(0);
		MFnDagNode fp(oparent);
		MDagPath pp;
		fp.getPath(pp);
		AddTransform(pp, file, beheadName);
	}
	
	std::string nodeName = path.fullPathName().asChar();
	if(beheadName.size() > 1) SHelper::behead(nodeName, beheadName);
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

bool HesperisIO::WriteCurves(MDagPathArray & paths, HesperisFile * file, const std::string & parentName) 
{
    CurveGroup gcurve;
    if(!CreateCurveGroup(paths, &gcurve)) {
        MGlobal::displayInfo(" hesperis check curves error");
        return false;
    }
    
	file->setWriteComponent(HesperisFile::WCurve);
    if(parentName.size()>1) {
        std::stringstream sst;
        sst<<parentName<<"|curves";
        MGlobal::displayInfo(MString("hes io write ")+sst.str().c_str());
        file->addCurve(sst.str(), &gcurve);
    }
    else
        file->addCurve("|curves", &gcurve);
	file->setDirty();
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save curves to file ")+ file->fileName().c_str());
	file->close();
	
	return true;
}

bool HesperisIO::WriteMeshes(MDagPathArray & paths, HesperisFile * file)
{
	MStatus stat;
	const unsigned n = paths.length();
	unsigned i, j;
	int numPnts = 0;
	unsigned numNodes = 0;
	unsigned numTris = 0;
	
	MGlobal::displayInfo(" hesperis check meshes");
	
	MIntArray triangleCounts, triangleVertices;
	MPointArray ps;
    MPoint wp;
	MMatrix worldTm;
	std::vector<ATriangleMesh * > meshes;
	for(i=0; i< n; i++) {
		MFnMesh fmesh(paths[i].node(), &stat);
		if(!stat) continue;
		numNodes++;
        
        worldTm = GetWorldTransform(paths[i]);
		
		numPnts = fmesh.numVertices();
		fmesh.getTriangles(triangleCounts, triangleVertices);
		numTris = triangleVertices.length() / 3;
		
		MGlobal::displayInfo(paths[i].fullPathName());
		MGlobal::displayInfo(MString(" vertex count: ") + numPnts);
		MGlobal::displayInfo(MString(" triangle count: ") + numTris);
	
		ATriangleMesh * amesh = new ATriangleMesh;
		meshes.push_back(amesh);
		amesh->create(numPnts, numTris);
		
		Vector3F * pnts = amesh->points();
		unsigned * inds = amesh->indices();
	
		fmesh.getPoints(ps, MSpace::kObject);
			
		for(j=0; j<numPnts; j++) {
            wp = ps[j] * worldTm;
			pnts[j].set((float)wp.x, (float)wp.y, (float)wp.z);
        }
		
		for(j=0; j<triangleVertices.length(); j++)
			inds[j] = triangleVertices[j];
			
		amesh->setDagName(std::string(paths[i].fullPathName().asChar()));
		std::stringstream sst;
		sst.str("");
		sst<<"mesh"<<numNodes;
		std::string meshName = sst.str();
		file->addTriangleMesh(meshName, amesh);
	}
	
	file->setDirty();
	file->setWriteComponent(HesperisFile::WTri);
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save mesh to file ")+ file->fileName().c_str());
	file->close();
	
	std::vector<ATriangleMesh * >::iterator it = meshes.begin();
	for(;it!=meshes.end();++it) delete *it;
	meshes.clear();
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

bool HesperisIO::GetCurves(const MDagPath &root, MDagPathArray & dst)
{
    MStatus stat;
	MItDag iter;
	iter.reset(root, MItDag::kDepthFirst, MFn::kNurbsCurve);
	for(; !iter.isDone(); iter.next()) {								
		MDagPath apath;		
		iter.getPath( apath );
		if(IsCurveValid(apath)) {
            MFnDagNode fdag(apath);
            if(!fdag.isIntermediateObject())
                dst.append(apath);
        }
	}
    return dst.length() > 0;
}

bool HesperisIO::ReadCurves(HesperisFile * file, MObject &target)
{
    MGlobal::displayInfo("opium read curve bundle");
    HWorld grpWorld;
    ReadTransforms(&grpWorld, target);
    ReadCurves(&grpWorld, target);
    grpWorld.close();
    return true;
}

bool HesperisIO::ReadTransforms(HBase * parent, MObject &target)
{
    std::vector<std::string > tmNames;
    parent->lsTypedChild<HTransform>(tmNames);
	std::vector<std::string>::const_iterator it = tmNames.begin();
	
    for(;it!=tmNames.end();++it) {
        std::string nodeName = *it;
        SHelper::behead(nodeName, parent->pathToObject());
        SHelper::behead(nodeName, "/");
        HBase child(*it);
        MObject otm = MObject::kNullObj;
        if(!FindNamedChild(otm, nodeName, target)) {
            MFnTransform ftransform;
            otm = ftransform.create(target);
            SHelper::noColon(nodeName);
            ftransform.setName(nodeName.c_str()); 
        }
        ReadTransforms(&child, otm);
        ReadCurves(&child, otm);
        child.close();
	}
    return true;
}

bool HesperisIO::ReadCurves(HBase * parent, MObject &target)
{
    std::vector<std::string > crvNames;
    parent->lsTypedChild<HCurveGroup>(crvNames);
	std::vector<std::string>::const_iterator it = crvNames.begin();
	
    for(;it!=crvNames.end();++it) {
        std::string nodeName = *it;
        SHelper::behead(nodeName, parent->pathToObject());
        SHelper::behead(nodeName, "/");
        HCurveGroup child(*it);
        MGlobal::displayInfo(MString(" create ") + nodeName.c_str());
        CurveGroup cg;
        child.load(&cg);
        CreateCurveGeos(&cg, target);
        child.close();
	}
    return true;
}

bool HesperisIO::CreateCurveGeos(CurveGroup * geos, MObject &target)
{
    if(CheckExistingCurves(geos, target)) return true;
    
    MGlobal::displayInfo(MString("create ")+ geos->numCurves()
    +MString(" curves"));
    
    Vector3F * pnts = geos->points();
    unsigned * cnts = geos->counts();
    unsigned i=0;
    for(;i<geos->numCurves();i++) {
        CreateACurve(pnts, cnts[i], target);
        pnts+= cnts[i];
    }
    return true;
}

bool HesperisIO::CreateACurve(Vector3F * pos, unsigned nv, MObject &target)
{
    MPointArray vertexArray;
    unsigned i=0;
	for(; i<nv; i++)
		vertexArray.append( MPoint( pos[i].x, pos[i].y, pos[i].z ) );
	const int degree = 2;
    const int spans = nv - degree;
	const int nknots = spans + 2 * degree - 1;
    MDoubleArray knotSequences;
	knotSequences.append(0.0);
	for(i = 0; i < nknots-2; i++)
		knotSequences.append( (double)i );
	knotSequences.append(nknots-3);
    
    MFnNurbsCurve curveFn;
	MStatus stat;
	curveFn.create(vertexArray,
					knotSequences, degree, 
					MFnNurbsCurve::kOpen, 
					false, false, 
					target, 
					&stat );
					
	return stat == MS::kSuccess;
}

bool HesperisIO::CheckExistingCurves(CurveGroup * geos, MObject &target)
{
    MDagPathArray existing;
    MDagPath root;
    MDagPath::getAPathTo(target, root);
    if(!GetCurves(root, existing)) return false;
    
    const unsigned ne = existing.length();
    if(ne != geos->numCurves()) return false;
    
    unsigned n = 0;
    unsigned i=0;
    for(;i<ne;i++) {
        MFnNurbsCurve fcurve(existing[i].node());
		n += fcurve.numCVs();
    }
    if(n!=geos->numPoints()) return false;
    
    MGlobal::displayInfo(" existing curves matched");
    
    return true;
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

bool HesperisIO::CreateCurveGroup(MDagPathArray & paths, CurveGroup * dst)
{
    MStatus stat;
	const unsigned n = paths.length();
	unsigned i, j;
	int numCvs = 0;
	unsigned numNodes = 0;
    
    for(i=0; i< n; i++) {
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
	for(i=0; i< n; i++) {
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

bool HesperisIO::LsCurves(std::vector<std::string > & dst)
{
    HWorld grpWorld;
    LsCurves(dst, &grpWorld);
    grpWorld.close();
    return true;   
}

bool HesperisIO::LsCurves(std::vector<std::string > & dst, HBase * parent)
{
    std::vector<std::string > tmNames;
    parent->lsTypedChild<HTransform>(tmNames);
	std::vector<std::string>::const_iterator ita = tmNames.begin();
	
    for(;ita!=tmNames.end();++ita) {
        HBase child(*ita);
        LsCurves(dst, &child);
        child.close();
	}
    
    std::vector<std::string > crvNames;
    parent->lsTypedChild<HCurveGroup>(crvNames);
	std::vector<std::string>::const_iterator itb = crvNames.begin();
	
    for(;itb!=crvNames.end();++itb) {
        dst.push_back(*itb);
	}
    return true;
}
//:~
