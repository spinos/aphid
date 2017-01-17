/*
 *  BoxPaintToolCmd.cpp
 *  proxyPaint
 *
 *  test pca
 *  select -r ball;
 *  float $mm2[] = `proxyPaintTool -pca -rto zxy`;
 *  xform -m $mm2[0] $mm2[1] $mm2[2] $mm2[3]
 *           $mm2[4] $mm2[5] $mm2[6] $mm2[7]
 *           $mm2[8] $mm2[9] $mm2[10] $mm2[11]
 *           $mm2[12] $mm2[13] $mm2[14] $mm2[15] box;
 *  select -r box;
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BoxPaintToolCmd.h"
#include <maya/MDagModifier.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MFnMesh.h>
#include "ProxyVizNode.h"
#include "ExampVizNode.h"
#include <AHelper.h>
#include <ASearchHelper.h>
#include <mama/ConnectionHelper.h>
#include <mama/SelectionHelper.h>
#include <geom/ATriangleMesh.h>
#include <geom/PrincipalComponents.h>
#include <kd/KdEngine.h>
#include <FieldTriangulation.h>

#define kBeginPickFlag "-bpk" 
#define kBeginPickFlagLong "-beginPick"
#define kDoPickFlag "-dpk" 
#define kDoPickFlagLong "-doPick"
#define kEndPickFlag "-epk" 
#define kEndPickFlagLong "-endPick"
#define kGetPickFlag "-gpk" 
#define kGetPickFlagLong "-getPick"
#define kConnectGroundFlag "-cgm" 
#define kConnectGroundFlagLong "-connectGMesh"
#define kSaveCacheFlag "-scf" 
#define kSaveCacheFlagLong "-saveCacheFile"
#define kLoadCacheFlag "-lcf" 
#define kLoadCacheFlagLong "-loadCacheFile"
#define kVoxFlag "-vxl" 
#define kVoxFlagLong "-voxelize"
#define kConnectVoxFlag "-cvx" 
#define kConnectVoxFlagLong "-connectVoxel"
#define kSelectVoxFlag "-svx" 
#define kSelectVoxFlagLong "-selectVox"
#define kPCAFlag "-pca" 
#define kPCAFlagLong "-principalComponent"
#define kRotateOrderPCAFlag "-rto" 
#define kRotateOrderPCAFlagLong "-rotateOrderPCA"
#define kDFTFlag "-dft" 
#define kDFTFlagLong "-distanceFieldTriangulate"
#define kDFTScaleFlag "-fts" 
#define kDFTScaleFlagLong "-fieldTriangulateScale"
#define kDFTRoundFlag "-ftr" 
#define kDFTRoundFlagLong "-fieldTriangulateRound"

#define kShrubCreateFlag "-csb" 
#define kShrubCreateFlagLong "-createShrub"

using namespace aphid;

proxyPaintTool::proxyPaintTool()
{
	setCommandString("proxyPaintToolCmd");
}

proxyPaintTool::~proxyPaintTool() {}

void* proxyPaintTool::creator()
{
	return new proxyPaintTool;
}

MSyntax proxyPaintTool::newSyntax()
{
	MSyntax syntax;

	syntax.addFlag(kBeginPickFlag, kBeginPickFlagLong, MSyntax::kString);
	syntax.addFlag(kDoPickFlag, kDoPickFlagLong, MSyntax::kString);
	syntax.addFlag(kEndPickFlag, kEndPickFlagLong, MSyntax::kString);
	syntax.addFlag(kEndPickFlag, kEndPickFlagLong, MSyntax::kString);
	syntax.addFlag(kGetPickFlag, kGetPickFlagLong, MSyntax::kString);
	syntax.addFlag(kSaveCacheFlag, kSaveCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kLoadCacheFlag, kLoadCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kConnectGroundFlag, kConnectGroundFlagLong);
	syntax.addFlag(kVoxFlag, kVoxFlagLong);
	syntax.addFlag(kConnectVoxFlag, kConnectVoxFlagLong);
	syntax.addFlag(kSelectVoxFlag, kSelectVoxFlagLong, MSyntax::kLong);
	syntax.addFlag(kPCAFlag, kPCAFlagLong);
	syntax.addFlag(kRotateOrderPCAFlag, kRotateOrderPCAFlagLong, MSyntax::kString);
	syntax.addFlag(kDFTFlag, kDFTFlagLong, MSyntax::kLong);
	syntax.addFlag(kDFTScaleFlag, kDFTScaleFlagLong, MSyntax::kDouble);
	syntax.addFlag(kDFTRoundFlag, kDFTRoundFlagLong, MSyntax::kDouble);
	syntax.addFlag(kShrubCreateFlag, kShrubCreateFlagLong, MSyntax::kNoArg);
	return syntax;
}

MStatus proxyPaintTool::doIt(const MArgList &args)
{
	MStatus status;

	status = parseArgs(args);
	
	if(m_operation == opUnknown) return status;
	
	if(m_operation == opConnectGround) return connectGroundSelected();
	
	if(m_operation == opSaveCache) return saveCacheSelected();
			
	if(m_operation == opLoadCache) return loadCacheSelected();
	
	if(m_operation == opVoxelize) return voxelizeSelected();
	
	if(m_operation == opConnectVoxel) return connectVoxelSelected();
	
    if(m_operation == opPrincipalComponent) return performPCA();
	
	if(m_operation == opDistanceFieldTriangulate) return performDFT();
	
	if(m_operation == opCreateShrub) {
		return creatShrub();
	}
		
	ASearchHelper finder;

	MObject oViz;
	if(!finder.getObjByFullName(fVizName.asChar(), oViz)) {
		MGlobal::displayWarning(MString("cannot find viz: ") + fVizName);
		return MS::kSuccess;
	}
	
	MFnDependencyNode fviz(oViz);
    ProxyViz *pViz = (ProxyViz*)fviz.userNode();
    
    if(!pViz) {
		MGlobal::displayWarning(MString("cannot recognize viz: ") + fVizName);
		return MS::kSuccess;
	}
	
	switch(m_operation) {
		case opBeginPick:
			pViz->beginPickInView();
			break;
		case opDoPick:
			pViz->processPickInView(m_currentVoxInd);
			break;
		case opEndPick:
			pViz->endPickInView();
			break;
		case opGetPick:
			setResult((int)pViz->numActivePlants() );
			break;
		default:
			;
	}

	return MS::kSuccess;
}

MStatus proxyPaintTool::parseArgs(const MArgList &args)
{
	m_operation = opUnknown;
    m_currentVoxInd = 0;
	m_rotPca = Matrix33F::XYZ;
	m_dftScale = 1;
	m_dftRound = 0;
	
	MStatus status;
	MArgDatabase argData(syntax(), args);

	if (argData.isFlagSet(kBeginPickFlag)) {
		status = argData.getFlagArgument(kBeginPickFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
		m_operation = opBeginPick;
	}
    
    if (argData.isFlagSet(kSelectVoxFlag)) {
		status = argData.getFlagArgument(kSelectVoxFlag, 0, m_currentVoxInd);
		if (!status) {
			status.perror("-selectVox flag parsing failed");
			return status;
		}
		AHelper::Info<int>(" proxyPaintTool select example", m_currentVoxInd);
	}
	
	if (argData.isFlagSet(kDoPickFlag)) {
		status = argData.getFlagArgument(kDoPickFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
		m_operation = opDoPick;
	}
	
	if (argData.isFlagSet(kEndPickFlag)) {
		status = argData.getFlagArgument(kEndPickFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
		m_operation = opEndPick;
	}
	
	if (argData.isFlagSet(kGetPickFlag)) {
		status = argData.getFlagArgument(kGetPickFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
		m_operation = opGetPick;
	}
	
	if (argData.isFlagSet(kConnectGroundFlag))
		m_operation = opConnectGround;
	
	if (argData.isFlagSet(kVoxFlag))
		m_operation = opVoxelize;
		
	if (argData.isFlagSet(kConnectVoxFlag))
		m_operation = opConnectVoxel;
	
	if (argData.isFlagSet(kSaveCacheFlag)) {
		status = argData.getFlagArgument(kSaveCacheFlag, 0, m_cacheName);
		if (!status) {
			status.perror("save cache flag parsing failed");
			return status;
		}
		m_operation = opSaveCache;
	}
	
	if (argData.isFlagSet(kLoadCacheFlag)) {
		status = argData.getFlagArgument(kLoadCacheFlag, 0, m_cacheName);
		if (!status) {
			status.perror("save cache flag parsing failed");
			return status;
		}
		m_operation = opLoadCache;
	}
    
    if (argData.isFlagSet(kPCAFlag)) {
		m_operation = opPrincipalComponent;
	}
	
	if (argData.isFlagSet(kRotateOrderPCAFlag)) {
		MString srod;
		status = argData.getFlagArgument(kRotateOrderPCAFlag, 0, srod);
		if (!status) {
			status.perror("pca rotate order -rto flag parsing failed");
			return status;
		}
		strToRotateOrder(srod);
	}
	
	if (argData.isFlagSet(kDFTFlag)) {
		status = argData.getFlagArgument(kDFTFlag, 0, m_dftLevel);
		if (!status) {
			status.perror("-dft flag parsing failed");
			return status;
		}
		m_operation = opDistanceFieldTriangulate;
	}
	
	if(argData.isFlagSet(kDFTScaleFlag)) {
		status = argData.getFlagArgument(kDFTScaleFlag, 0, m_dftScale);
		if (!status) {
			MGlobal::displayWarning(" proxyPaintTool cannot parse -fts flag");
		}
	}
	
	if(argData.isFlagSet(kDFTRoundFlag)) {
		status = argData.getFlagArgument(kDFTRoundFlag, 0, m_dftRound);
		if (!status) {
			MGlobal::displayWarning(" proxyPaintTool cannot parse -ftr flag");
		}
	}
	
	if(argData.isFlagSet(kShrubCreateFlag)) {
		m_operation = opCreateShrub;
	}
	
	return MS::kSuccess;
}

MStatus proxyPaintTool::finalize()
{
	MArgList command;
	command.addArg(commandString());
	return MPxToolCommand::doFinalize( command );
}

MStatus proxyPaintTool::connectGroundSelected()
{
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 2) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select mesh(es) and a viz to connect");
		return MS::kFailure;
	}
	
	MStatus stat;
	MObject vizobj = SelectionHelper::GetTypedNode(sels, "proxyViz", MFn::kPluginLocatorNode);
	if(vizobj == MObject::kNullObj ) {
		MGlobal::displayWarning("proxyPaintTool found no viz selected, select mesh(es) and a viz to connect");
		return MS::kFailure;
	}
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
	
	MItSelectionList meshIter(sels, MFn::kMesh, &stat);
	if(!stat) {
		MGlobal::displayWarning("proxyPaintTool no mesh selected");
		return MS::kFailure;
	}
	
	ProxyViz* pViz = (ProxyViz*)fviz.userNode();
	pViz->setEnableCompute(false);
	
	bool connStat;
	
	for(;!meshIter.isDone(); meshIter.next() ) {
		MObject meshobj;
		meshIter.getDependNode(meshobj);
		
		MDagPath transPath;
		meshIter.getDagPath ( transPath );
		transPath.pop();
		
		MPlug worldSpacePlug;
		int slotI = -1;
		if(isTransformConnected(transPath, vizobj, slotI, worldSpacePlug) ) {
			AHelper::Info<MString>(" WARNING skip connected transform", transPath.fullPathName() );
			if(isMeshConnectedSlot(meshobj, vizobj, slotI) ) {
				continue;
			}
		}
		
		connStat = ConnectionHelper::ConnectToArray(meshobj, "outMesh", 
										vizobj, "groundMesh",
										slotI);

		if(connStat) {
			AHelper::Info<MString>("proxyPaintTool connect ground", transPath.fullPathName() );
			connectTransform(worldSpacePlug, vizobj);
			checkOutputConnection(vizobj, "ov");
		}
	}
	
	pViz->setEnableCompute(true);
	
	return stat;
}

MStatus proxyPaintTool::connectVoxelSelected()
{
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 2) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select proxyExample(s) and a viz to connect");
		return MS::kFailure;
	}
	
	MStatus stat;
	MObject vizobj = SelectionHelper::GetTypedNode(sels, "proxyViz", MFn::kPluginLocatorNode);
	if(vizobj == MObject::kNullObj ) {
		return MS::kFailure;
	}
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
	
	ProxyViz* pViz = (ProxyViz*)fviz.userNode();
	pViz->setEnableCompute(false);
	
	MItSelectionList voxIter(sels, MFn::kPluginLocatorNode, &stat);
	
	for(;!voxIter.isDone(); voxIter.next() ) {
		MObject vox;
		voxIter.getDependNode(vox);
		
		MFnDependencyNode fvox(vox, &stat);
		if(!stat) continue;
		
		if(fvox.typeName() != "proxyExample"
			&& fvox.typeName() != "shrubViz") continue;
			
		if(connectVoxToViz(vox, vizobj) ) {
			AHelper::Info<MString>("proxyPaintTool connect (bundle) example", fvox.name() );
			checkOutputConnection(vizobj, "ov1");
		}
	}
	
	pViz->setEnableCompute(true);
	
	return stat;
}

bool proxyPaintTool::connectVoxToViz(MObject & voxObj, MObject & vizObj)
{
	MFnDependencyNode fvox(voxObj);
	MPlug srcPlug = fvox.findPlug("ov");
	if(ConnectionHelper::ConnectedToNode(srcPlug, vizObj) ) {
		return false;
	}	
	ConnectionHelper::ConnectToArray(voxObj, "ov", vizObj, "ixmp");
	return true;
}

void proxyPaintTool::connectTransform(MPlug & worldSpacePlug, MObject & vizObj)
{
	if(ConnectionHelper::ConnectedToNode(worldSpacePlug, vizObj) ) {
		return;
	}
	ConnectionHelper::ConnectToArray(worldSpacePlug, vizObj, "groundSpace");
}

MStatus proxyPaintTool::saveCacheSelected()
{
	MStatus stat;
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select a viz to save cache");
		return MS::kFailure;
	}
	
	MObject vizobj = SelectionHelper::GetTypedNode(sels, "proxyViz", MFn::kPluginLocatorNode);
	if(vizobj == MObject::kNullObj ) {
		return MS::kFailure;
	}
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
		
	ProxyViz* pViz = (ProxyViz*)fviz.userNode();
	pViz->saveExternal(m_cacheName.asChar() );
	
	return stat;
}
	
MStatus proxyPaintTool::loadCacheSelected()
{
	MStatus stat;
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select a viz to load cache");
		return MS::kFailure;
	}
	
	MObject vizobj = SelectionHelper::GetTypedNode(sels, "proxyViz", MFn::kPluginLocatorNode);
	if(vizobj == MObject::kNullObj ) {
		return MS::kFailure;
	}
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
		
	ProxyViz* pViz = (ProxyViz*)fviz.userNode();
	pViz->loadExternal(m_cacheName.asChar() );
	
	return stat;
}

MStatus proxyPaintTool::voxelizeSelected()
{
	MStatus stat;
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select mesh(es) to voxelize");
		return MS::kFailure;
	}
	
	MObject vizobj = SelectionHelper::GetTypedNode(sels, "proxyExample", MFn::kPluginLocatorNode);
	if(vizobj == MObject::kNullObj ) {
		vizobj = AHelper::CreateDagNode("proxyExample", "proxyExample");
		if(vizobj.isNull()) {
			MGlobal::displayWarning("proxyPaintTool cannot create example viz");
			return MS::kFailure;
		}
	}
	
	MItSelectionList meshIter(sels, MFn::kMesh, &stat);
	if(!stat) {
		MGlobal::displayWarning("proxyPaintTool no mesh selected");
		return MS::kFailure;
	}
	
	sdb::VectorArray<cvx::Triangle> tris;
	BoundingBox bbox;
	
	for(;!meshIter.isDone(); meshIter.next() ) {
		
		MDagPath meshPath;
		meshIter.getDagPath(meshPath);
		getMeshTris(tris, bbox, meshPath);
	}
	
	if(tris.size() < 1) {
		MGlobal::displayWarning("proxyPaintTool no triangle added");
		return MS::kFailure;
	}
	
	bbox.round();
	AHelper::Info<unsigned>("proxyPaintTool voxelize n triangle", tris.size() );

	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool init viz node", fviz.name() );
		
	ExampViz* pViz = (ExampViz*)fviz.userNode();
	pViz->voxelize2(&tris, bbox);
	
	return stat;
}

void proxyPaintTool::getMeshTris(sdb::VectorArray<cvx::Triangle> & tris,
								BoundingBox & bbox,
								const MDagPath & meshPath)
{
	AHelper::Info<MString>("get mesh triangles", meshPath.fullPathName() );
	
	MMatrix worldTm = AHelper::GetWorldParentTransformMatrix(meshPath);
	
    MStatus stat;
	
    MIntArray vertices;
    int i, j, nv;
	MPoint dp[3];
	Vector3F fp[3];
	MItMeshPolygon faceIt(meshPath);
    for(; !faceIt.isDone(); faceIt.next() ) {

		faceIt.getVertices(vertices);
        nv = vertices.length();
        
        for(i=1; i<nv-1; ++i ) {
			dp[0] = faceIt.point(0, MSpace::kObject );
			dp[1] = faceIt.point(i, MSpace::kObject );
			dp[2] = faceIt.point(i+1, MSpace::kObject );
			
			dp[0] *= worldTm;
			dp[1] *= worldTm;	
			dp[2] *= worldTm;
			
			cvx::Triangle tri;
			for(j=0; j<3; ++j) {
				fp[j].set(dp[j].x, dp[j].y, dp[j].z);
				tri.setP(fp[j], j);
				bbox.expandBy(fp[j], 1e-4f);
			}
			
			tris.insert(tri);
        }
    }
}

void proxyPaintTool::checkOutputConnection(MObject & node, const MString & outName)
{
	MFnDependencyNode fnode(node);
	MStatus stat;
	MPlug outPlug = fnode.findPlug(outName, &stat);
	if(!stat) {
		AHelper::Info<MString>(" proxyPaintTool error not named plug", outName);
		return;
	}
	
	if(outPlug.isConnected() ) return;
	
	MDagModifier modif;
	MObject trans = modif.createNode("transform");
	modif.renameNode (trans, fnode.name() + "_" + outName);
	modif.doIt();
	
	modif.connect(outPlug, MFnDependencyNode(trans).findPlug("tx") );
	modif.doIt();
	
	MFnDagNode fdag(node);
	MDagPath parentPath;
	fdag.getPath(parentPath);
	parentPath.pop();
	
	modif.reparentNode(trans, parentPath.node() );
	modif.doIt();
}

MStatus proxyPaintTool::performPCA()
{
    MStatus stat;
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select mesh(es) to pca");
		return MS::kFailure;
	}

	MItSelectionList meshIter(sels, MFn::kMesh, &stat);
	if(!stat) {
		MGlobal::displayWarning("proxyPaintTool no mesh selected");
		return MS::kFailure;
	}
	
	sdb::VectorArray<cvx::Triangle> tris;
	BoundingBox bbox;
	
	for(;!meshIter.isDone(); meshIter.next() ) {
		
		MDagPath meshPath;
		meshIter.getDagPath(meshPath);
		getMeshTris(tris, bbox, meshPath);
	}
	
	if(tris.size() < 1) {
		MGlobal::displayWarning("proxyPaintTool no triangle added");
		return MS::kFailure;
	}
	
	bbox.round();
    
    const int nt = tris.size();
    AHelper::Info<int>("proxyPaintTool pca n triangles", nt );
    std::vector<Vector3F> pnts;
    for(int i=0; i< nt; ++i) {
        const cvx::Triangle * t = tris[i];
/// at triangle center
        pnts.push_back(t->P(0) * .33f 
                       + t->P(1) * .33f
                       + t->P(2) * .33f);
    }
    
    PrincipalComponents<std::vector<Vector3F> > obpca;
    AOrientedBox obox = obpca.analyze(pnts, pnts.size(), m_rotPca );
	
    AHelper::Info<int>("pca order", m_rotPca );
    AHelper::Info<Vector3F>("obox c", obox.center() );
    AHelper::Info<Matrix33F>("obox r", obox.orientation() );
    AHelper::Info<Vector3F>("obox e", obox.extent() );
    MDoubleArray rd;
    rd.setLength(16);
    const Vector3F exob = obox.extent();
    const Matrix33F rtob = obox.orientation();
/// scale to unit box
    Vector3F rx = obox.orientation().row(0) * exob.x * 2.f;
    
    rd[0] = rx.x;
    rd[1] = rx.y;
    rd[2] = rx.z;
    rd[3] = 0.f; 
    
    Vector3F ry = obox.orientation().row(1) * exob.y * 2.f;
    rd[4] = ry.x;
    rd[5] = ry.y;
    rd[6] = ry.z;
    rd[7] = 0.f;
    
    Vector3F rz = obox.orientation().row(2) * exob.z * 2.f;
    rd[8] = rz.x;
    rd[9] = rz.y;
    rd[10] = rz.z;
    rd[11] = 0.f;
    
    rd[12] = obox.center().x;
    rd[13] = obox.center().y;
    rd[14] = obox.center().z;
    rd[15] = 1.f;
    setResult(rd);
	return stat;
}

void proxyPaintTool::strToRotateOrder(const MString & srod)
{
	if(srod == "yzx") m_rotPca = Matrix33F::YZX;
	else if(srod == "zxy") m_rotPca = Matrix33F::ZXY;
	else if(srod == "xzy") m_rotPca = Matrix33F::XZY;
	else if(srod == "yxz") m_rotPca = Matrix33F::YXZ;
	else if(srod == "zyx") m_rotPca = Matrix33F::ZYX;
}

MStatus proxyPaintTool::performDFT()
{
	MStatus stat;
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select mesh(es) to reduce");
		return MS::kFailure;
	}
	
	MItSelectionList meshIter(sels, MFn::kMesh, &stat);
	if(!stat) {
		MGlobal::displayWarning("proxyPaintTool no mesh selected");
		return MS::kFailure;
	}
	
	sdb::VectorArray<cvx::Triangle> tris;
	BoundingBox bbox;
	
	for(;!meshIter.isDone(); meshIter.next() ) {
		
		MDagPath meshPath;
		meshIter.getDagPath(meshPath);
		getMeshTris(tris, bbox, meshPath);
	}
	
	if(tris.size() < 1) {
		MGlobal::displayWarning("proxyPaintTool no triangle added");
		return MS::kFailure;
	}
	
	bbox.round();
	AHelper::Info<unsigned>("proxyPaintTool reducing n triangle", tris.size() );

	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 64;
	KdEngine engine;
	KdNTree<cvx::Triangle, KdNode4 > gtr;
	engine.buildTree<cvx::Triangle, KdNode4, 4>(&gtr, &tris, bbox, &bf);
	
	float uscale = m_dftScale;
	if(uscale < 1.f) {
		MGlobal::displayInfo(" proxyPaintTool dft scale truncate to 1");
		uscale = 1.f;
	}
	else if(uscale > 4.f) {
		MGlobal::displayInfo(" proxyPaintTool dft scale truncate to 2");
		uscale = 4.f;
	}
	
	float uround = m_dftRound;
	if(uround < 0.f) {
		MGlobal::displayInfo(" proxyPaintTool dft roundness truncate to 0");
		uround = 1.f;
	}
	else if(uround > .5f) {
		MGlobal::displayInfo(" proxyPaintTool dft roundness truncate to 0.5");
		uround = .5f;
	}
	
	BoundingBox tb = gtr.getBBox();
	const float gz = tb.getLongestDistance() * 1.01f * uscale;
	const Vector3F cent = tb.center();
	tb.setMin(cent.x - gz, cent.y - gz, cent.z - gz );
	tb.setMax(cent.x + gz, cent.y + gz, cent.z + gz );
	
	ttg::FieldTriangulation msh;
	msh.fillBox(tb, gz);
	
	BDistanceFunction distFunc;
	distFunc.addTree(&gtr);
	distFunc.setDomainDistanceRange(gz * GDT_FAC_ONEOVER8 );
	
	int bLevel = m_dftLevel;
	if(bLevel < 3) {
		AHelper::Info<int>("proxyPaintTool reducing level < ", 3 );
		bLevel = 3;
	}
	else if(bLevel > 6) {
		AHelper::Info<int>("proxyPaintTool reducing level > ", 6 );
		bLevel = 6;
	}
	
	msh.frontAdaptiveBuild<BDistanceFunction>(&distFunc, 3, bLevel, .47f + uround );
	msh.triangulateFront();
	AHelper::Info<int>(" proxyPaintTool reduce selected to n triangle ", msh.numFrontTriangles() );
	AHelper::Info<int>(" n vertex ", msh.numTriangleVertices() );

	const int numVertex = msh.numTriangleVertices();
	const int numPolygon = msh.numFrontTriangles();
	MPointArray vertexArray;
	int i;
	for(i=0; i<numVertex;++i) {
		const Vector3F & p = msh.triangleVertexP()[i];
		vertexArray.append(MPoint(p.x, p.y, p.z) );
	}
	
	MIntArray polygonCounts;
	MIntArray polygonConnects;
	
	for(i=0; i<numPolygon;++i) {
		polygonCounts.append(3);
		polygonConnects.append(msh.triangleIndices()[i*3]);
		polygonConnects.append(msh.triangleIndices()[i*3+1]);
		polygonConnects.append(msh.triangleIndices()[i*3+2]);
	}
	
	MFnMesh meshFn;
	meshFn.create( numVertex, numPolygon, vertexArray, polygonCounts, polygonConnects, MObject::kNullObj, &stat );
	
	return stat;
}

bool proxyPaintTool::isTransformConnected(const MDagPath & transPath, 
								const MObject & vizObj,
								int & slotPhyInd,
								MPlug & worldSpacePlug)
{
	MFnDependencyNode ftrans(transPath.node() );
	unsigned wsi = 0;
	if(transPath.isInstanced() ) {
		wsi = transPath.instanceNumber();
		AHelper::Info<MString>(" instanced transform", transPath.fullPathName() );
		AHelper::Info<unsigned>(" world matrix space id", wsi );
	}
	worldSpacePlug = ftrans.findPlug("worldMatrix").elementByLogicalIndex(wsi);
	return ConnectionHelper::ConnectedToNode(worldSpacePlug, vizObj, &slotPhyInd);
}

bool proxyPaintTool::isMeshConnectedSlot(const MObject & meshObj, 
					const MObject & vizObj,
					const int & slotPhyInd)
{
	MFnDependencyNode fnode(meshObj);
	MPlug srcPlug = fnode.findPlug("outMesh");
	
	int inSlot = -1;
	if(ConnectionHelper::ConnectedToNode(srcPlug, vizObj, &inSlot) ) {
		return inSlot == slotPhyInd;
	}
	
	return false;
}
//:~