#include "boxPaintTool.h"
#include <maya/MFnCamera.h>
#include <maya/MPointArray.h>
#include <maya/MDagModifier.h>
#include "../core/TriangleRaster.h"
#include <ASearchHelper.h>

#define kOptFlag "-opt" 
#define kOptFlagLong "-option"
#define kNsegFlag "-nsg" 
#define kNsegFlagLong "-numSegment"
#define kLsegFlag "-brd" 
#define kLsegFlagLong "-brushRadius"
#define kMinFlag "-smn" 
#define kMinFlagLong "-scaleMin"
#define kMaxFlag "-smx" 
#define kMaxFlagLong "-scaleMax"
#define kRotateNoiseFlag "-rno" 
#define kRotateNoiseFlagLong "-rotateNoise"
#define kWeightFlag "-bwt" 
#define kWeightFlagLong "-brushWeight"
#define kNormalFlag "-anl" 
#define kNormalFlagLong "-alongNormal"
#define kWriteCacheFlag "-wch" 
#define kWriteCacheFlagLong "-writeCache"
#define kReadCacheFlag "-rch" 
#define kReadCacheFlagLong "-readCache"
#define kCullSelectionFlag "-cus" 
#define kCullSelectionFlagLong "-cullSelection"
#define kMultiCreateFlag "-mct" 
#define kMultiCreateFlagLong "-multiCreate"
#define kInstanceGroupCountFlag "-igc" 
#define kInstanceGroupCountFlagLong "-instanceGroupCount"
#define kBlockFlag "-bl" 
#define kBlockFlagLong "-block"
#define kVizFlag "-v" 
#define kVizFlagLong "-visualizer"

proxyPaintTool::~proxyPaintTool() {}

proxyPaintTool::proxyPaintTool()
{
	setCommandString("proxyPaintToolCmd");
}

void* proxyPaintTool::creator()
{
	return new proxyPaintTool;
}

MSyntax proxyPaintTool::newSyntax()
{
	MSyntax syntax;

	syntax.addFlag(kOptFlag, kOptFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kNsegFlag, kNsegFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kWeightFlag, kWeightFlagLong, MSyntax::kDouble );
	syntax.addFlag(kLsegFlag, kLsegFlagLong, MSyntax::kDouble );
	syntax.addFlag(kMinFlag, kMinFlagLong, MSyntax::kDouble );
	syntax.addFlag(kMaxFlag, kMaxFlagLong, MSyntax::kDouble );
	syntax.addFlag(kRotateNoiseFlag, kRotateNoiseFlagLong, MSyntax::kDouble );
	syntax.addFlag(kNormalFlag, kNormalFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kWriteCacheFlag, kWriteCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kReadCacheFlag, kReadCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kCullSelectionFlag, kCullSelectionFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kMultiCreateFlag, kMultiCreateFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kInstanceGroupCountFlag, kInstanceGroupCountFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kBlockFlag, kBlockFlagLong, MSyntax::kString);
	syntax.addFlag(kVizFlag, kVizFlagLong, MSyntax::kString);
	
	return syntax;
}

MStatus proxyPaintTool::doIt(const MArgList &args)
//
// Description
//     Sets up the helix parameters from arguments passed to the
//     MEL command.
//
{
	MStatus status;

	status = parseArgs(args);
	
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
	
	MDagPath pMesh;
	if(finder.dagByFullName(fBlockerName.asChar(), pMesh)) {
		MGlobal::displayInfo(MString("found blocker mesh: ") + fBlockerName);
		pViz->setCullMesh(pMesh);
	}
	else {
		MDagPath nouse;
		pViz->setCullMesh(nouse);
	}

	return MS::kSuccess;
}

MStatus proxyPaintTool::parseArgs(const MArgList &args)
{
	MStatus status;
	MArgDatabase argData(syntax(), args);
	
	if (argData.isFlagSet(kOptFlag)) {
		unsigned tmp;
		status = argData.getFlagArgument(kOptFlag, 0, tmp);
		if (!status) {
			status.perror("opt flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kNsegFlag)) {
		unsigned tmp;
		status = argData.getFlagArgument(kNsegFlag, 0, tmp);
		if (!status) {
			status.perror("numseg flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kLsegFlag)) {
		double tmp;
		status = argData.getFlagArgument(kLsegFlag, 0, tmp);
		if (!status) {
			status.perror("lenseg flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kWeightFlag)) {
		double tmp;
		status = argData.getFlagArgument(kWeightFlag, 0, tmp);
		if (!status) {
			status.perror("weight flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kNormalFlag)) {
		unsigned aln;
		status = argData.getFlagArgument(kNormalFlag, 0, aln);
		if (!status) {
			status.perror("normal flag parsing failed");
			return status;
		}
	}

	if (argData.isFlagSet(kWriteCacheFlag)) 
	{
		MString ch;
		status = argData.getFlagArgument(kWriteCacheFlag, 0, ch);
		if (!status) {
			status.perror("cache out flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kReadCacheFlag)) 
	{
		MString ch;
		status = argData.getFlagArgument(kReadCacheFlag, 0, ch);
		if (!status) {
			status.perror("cache in flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kMinFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kMinFlag, 0, noi);
		if (!status) {
			status.perror("min flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kMaxFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kMaxFlag, 0, noi);
		if (!status) {
			status.perror("max flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kRotateNoiseFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kRotateNoiseFlag, 0, noi);
		if (!status) {
			status.perror("rotate noise flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kCullSelectionFlag)) {
		unsigned cus;
		status = argData.getFlagArgument(kCullSelectionFlag, 0, cus);
		if (!status) {
			status.perror("cull selection flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kMultiCreateFlag)) {
		unsigned mcr;
		status = argData.getFlagArgument(kMultiCreateFlag, 0, mcr);
		if (!status) {
			status.perror("multi create flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kInstanceGroupCountFlag)) {
		unsigned igc;
		status = argData.getFlagArgument(kInstanceGroupCountFlag, 0, igc);
		if (!status) {
			status.perror("instance group count flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kBlockFlag)) {
		status = argData.getFlagArgument(kBlockFlag, 0, fBlockerName);
		if (!status) {
			status.perror("block flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kVizFlag)) {
		status = argData.getFlagArgument(kVizFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
	}
	
	MGlobal::displayInfo(MString("culled by ") + fBlockerName + " " + fVizName);
	
	return MS::kSuccess;
}

MStatus proxyPaintTool::finalize()
//
// Description
//     Command is finished, construct a string for the command
//     for journalling.
//
{
	MArgList command;
	command.addArg(commandString());
	//command.addArg(MString(kOptFlag));
	//command.addArg((int)opt);
	//command.addArg(MString(kNSegFlag));
	//command.addArg((int)nseg);
	//command.addArg(MString(kLSegFlag));
	//command.addArg((float)lseg);
	return MPxToolCommand::doFinalize( command );
}

proxyPaintContext::proxyPaintContext():mOpt(999),m_numSeg(5),m_brushRadius(4.f),m_brushWeight(.66f),m_min_scale(1.f),m_max_scale(1.f),m_rotation_noise(0.f),m_pViz(0),
m_growAlongNormal(0),
m_cullSelection(0), 
m_multiCreate(0),
m_groupCount(1)
{
	setTitleString ( "proxyPaint Tool" );

	// Tell the context which XPM to use so the tool can properly
	// be a candidate for the 6th position on the mini-bar.
	setImage("proxyPaintTool.xpm", MPxContext::kImage1 );
	setOperation(2);
}

void proxyPaintContext::toolOnSetup ( MEvent & )
{
	setHelpString( helpString );
}

MStatus proxyPaintContext::doPress( MEvent & event )
{
	m_listAdjustment = MGlobal::kAddToList;
	if ( event.isModifierControl() )
		m_listAdjustment = MGlobal::kRemoveFromList;

	event.getPosition( start_x, start_y );
	
	view = M3dView::active3dView();

/// get camera matrix
	MDagPath camera;
	view.getCamera( camera );
	
	MFnCamera fnCamera( camera );
	MVector upDir = fnCamera.upDirection( MSpace::kWorld );
	MVector rightDir = fnCamera.rightDirection( MSpace::kWorld );
	MVector viewDir = fnCamera.viewDirection( MSpace::kWorld );
	MPoint eyePos = fnCamera.eyePoint ( MSpace::kWorld );
	_worldEye = eyePos;
	
	clipNear = fnCamera.nearClippingPlane();
	clipFar = fnCamera.farClippingPlane();
	
	mat.setIdentity ();
	*mat.m(0, 0) = -rightDir.x;
	*mat.m(0, 1) = -rightDir.y;
	*mat.m(0, 2) = -rightDir.z;
	*mat.m(1, 0) = upDir.x;
	*mat.m(1, 1) = upDir.y;
	*mat.m(1, 2) = upDir.z;
	*mat.m(2, 0) = viewDir.x;
	*mat.m(2, 1) = viewDir.y;
	*mat.m(2, 2) = viewDir.z;
	*mat.m(3, 0) = eyePos.x;
	*mat.m(3, 1) = eyePos.y;
	*mat.m(3, 2) = eyePos.z;
	
	mat.inverse();

	validateSelection();

	_seed = rand() % 999999;
	
	if(mOpt == 2) startProcessSelect();
	if(mOpt == 9) startSelectGround();
	
	return MS::kSuccess;		
}


MStatus proxyPaintContext::doDrag( MEvent & event )
//
// Drag out the proxyPaint (using OpenGL)
//
{
	event.getPosition( last_x, last_y );

	switch (mOpt)
	{
		case 0 :
			grow();
			break;
		case 1 :
			erase();
			break;
		case 2 : 
			processSelect();
			break;
		case 3 :
			resize();
			break;
		case 4 :
			move();
			break;
		case 5 :
			rotateAroundAxis(1);
			break;
		case 6 :
			rotateAroundAxis(2);
			break;
		case 7 :
			rotateAroundAxis(0);
			break;
		case 8 :
			//moveAlongAxis(1);
			break;
		case 9 :
			selectGround();
			break;
		default:
			;
	}

	start_x = last_x;
	start_y = last_y;
	
	view.refresh( true );
	return MS::kSuccess;		
}

MStatus proxyPaintContext::doRelease( MEvent & event )
{	
	event.getPosition( last_x, last_y );
	
	if(!m_pViz) return MS::kSuccess;
	if(mOpt==0) m_pViz->finishGrow();
	if(mOpt==1) m_pViz->finishErase();
	if(mOpt==2) AHelper::Info<unsigned>("n active plants", m_pViz->numActivePlants() );
	if(mOpt==9) AHelper::Info<unsigned>("n active faces", m_pViz->numActiveGroundFaces() );
	
	return MS::kSuccess;		
}

MStatus proxyPaintContext::doEnterRegion( MEvent & )
{
	return setHelpString( helpString );
}

void proxyPaintContext::getClassName( MString & name ) const
{
	name.set("proxyPaint");
}

void proxyPaintContext::setOperation(unsigned val)
{
	if(val == 99) {
		cleanup();
		return;
	}
	
	if(val==100) {
		flood();
		return;
	}
	
	if(val==101) {
		snap();
		return;
	}
	
	if(val == 102) {
		extractSelected();
		return;
	}
	
	if(val == 103) {
		erectSelected();
		return;
	}
	
	if(mOpt == 9 && val != 9) {
		MGlobal::setSelectionMode(MGlobal::kSelectObjectMode);
		MSelectionList empty;
		MGlobal::selectCommand(empty);
	}
	
	mOpt = val;
	switch (mOpt)
	{
		case 0:
			MGlobal::displayInfo("proxyPaint set to create");
			break;
		case 1: 
			MGlobal::displayInfo("proxyPaint set to erase");
			break;
		case 2: 
			MGlobal::displayInfo("proxyPaint set to select");
			break;
		case 3:
			MGlobal::displayInfo("proxyPaint set to scale");
			break;
		case 4:
			MGlobal::displayInfo("proxyPaint set to move");
			break;
		case 5:
			MGlobal::displayInfo("proxyPaint set to rotate y");
			break;
		case 6:
			MGlobal::displayInfo("proxyPaint set to rotate z");
			break;
		case 7:
			MGlobal::displayInfo("proxyPaint set to rotate x");
			break;
		case 8:
			MGlobal::displayInfo("proxyPaint set to translate along y axis");
			break;
		case 9:
			MGlobal::displayInfo("proxyPaint set to select ground faces");
			break;
		case 10:
			MGlobal::displayInfo("proxyPaint set to smooth selected");
			break;
		default:
			;
	}
	
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getOperation() const
{
	return mOpt;
}

void proxyPaintContext::setNSegment(unsigned val)
{
	m_numSeg = val;
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getNSegment() const
{
	return m_numSeg;
}

void proxyPaintContext::setBrushRadius(float val)
{
	m_brushRadius = val;
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getBrushRadius() const
{
	return m_brushRadius;
}

void proxyPaintContext::setScaleMin(float val)
{
	m_min_scale = val;
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getScaleMin() const
{
	return m_min_scale;
}

void proxyPaintContext::setScaleMax(float val)
{
	m_max_scale = val;
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getScaleMax() const
{
	return m_max_scale;
}

void proxyPaintContext::setRotationNoise(float val)
{
	m_rotation_noise = val;
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getRotationNoise() const
{
	return m_rotation_noise;
}

void proxyPaintContext::setBrushWeight(float val)
{
	m_brushWeight = val;
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getBrushWeight() const
{
	return m_brushWeight;
}

void proxyPaintContext::setGrowAlongNormal(unsigned val)
{
	if(val == 1) 
		MGlobal::displayInfo("proxyPaint enable grow along face normal");
	else
		MGlobal::displayInfo("proxyPaint disable grow along face normal");
	m_growAlongNormal = val;
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getGrowAlongNormal() const
{
	return m_growAlongNormal;
}

void proxyPaintContext::setCullSelection(unsigned val)
{
	if(val == 1) 
		MGlobal::displayInfo("proxyPaint enable cull selection");
	else
		MGlobal::displayInfo("proxyPaint disable cull selection");
	m_cullSelection = val;
	sendCullSurface();
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getCullSelection() const
{
	return m_cullSelection;
}

void proxyPaintContext::setMultiCreate(unsigned val)
{
	if(val == 1) 
		MGlobal::displayInfo("proxyPaint enable multiple create");
	else
		MGlobal::displayInfo("proxyPaint disable multiple create");
	m_multiCreate = val;
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getMultiCreate() const
{
	return m_multiCreate;
}

void proxyPaintContext::setInstanceGroupCount(unsigned val)
{
	MGlobal::displayInfo(MString("proxyPaint will extract transforms into ") + val + " groups");
	m_groupCount = val;
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getInstanceGroupCount() const
{
	return m_groupCount;
}

void proxyPaintContext::resize()
{
	if(!m_pViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
		
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	
	m_pViz->adjustSize(fromNear, fromFar, mag);
}

void proxyPaintContext::move()
{
	if(!m_pViz) return;
		
	m_pViz->adjustPosition(start_x, start_y, last_x, last_y,  clipNear, clipFar, mat);
}

void proxyPaintContext::rotateAroundAxis(short axis)
{
	if(!m_pViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
		
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	
    m_pViz->setNoiseWeight(m_rotation_noise);
	m_pViz->adjustRotation(fromNear, fromFar, mag, axis);
}

void proxyPaintContext::moveAlongAxis(short axis)
{
	//if(!m_pViz) return;
	//m_pViz->adjustLocation(start_x, start_y, last_x, last_y,  clipNear, clipFar, mat, axis);
}

void proxyPaintContext::selectGround()
{
	if(!m_pViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	m_pViz->selectGround(fromNear, fromFar, m_listAdjustment);
}

void proxyPaintContext::startSelectGround()
{
	if(!m_pViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld (start_x, start_y, fromNear, fromFar );
	
	m_pViz->selectGround(fromNear, fromFar, MGlobal::kReplaceList);
}

void proxyPaintContext::smoothSelected()
{}

void proxyPaintContext::grow()
{
	if(!m_pViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	ProxyViz::GrowOption opt;
	setGrowOption(opt);
	m_pViz->grow(fromNear, fromFar, opt);
}

char proxyPaintContext::validateSelection()
{
	MSelectionList slist;
 	MGlobal::getActiveSelectionList( slist );
	if(!validateViz(slist))
	    MGlobal::displayWarning("No proxyViz selected");
		
	if(!validateCollide(slist))
		MGlobal::displayWarning("No mesh selected");
		
    if(!m_pViz)
		return 0;
	if(!goCollide)
		return 0;
		
	return 1;
}

void proxyPaintContext::flood()
{
	if(!m_pViz) return;
	ProxyViz::GrowOption opt;
	setGrowOption(opt);
	m_pViz->flood(opt);
}

void proxyPaintContext::extractSelected()
{
	if(!m_pViz)
		return;
		
	unsigned numSelected = m_pViz->getNumActiveBoxes();
	
	if(numSelected < 1) return;
	
	MDagModifier mod;
	MStatus stat;
	MObjectArray instanceGroups;
	
	for(int g = 0; g < m_groupCount; g++) {
		MObject grp = mod.createNode("transform", MObject::kNullObj , &stat);
		mod.doIt();
		instanceGroups.append(grp);
		MFnDagNode fgrp(grp);
		fgrp.setName(MString("instanceGroup")+g);
	}

	PseudoNoise pnoise;	
	for(unsigned i = 0; i < numSelected; i++) {
		const MMatrix mat = m_pViz->getActiveBox(i);
		const int idx =  m_pViz->getActiveIndex(i);
		const int groupId = pnoise.rint1(idx + 2397 * idx, m_groupCount * 4) % m_groupCount;
		MObject tra = mod.createNode("transform", instanceGroups[groupId], &stat);
		mod.doIt();
		MFnTransform ftra(tra);
		ftra.set(MTransformationMatrix(mat));
		ftra.setName(MString("transform") + idx);
		MObject loc = mod.createNode("locator", tra, &stat);
		mod.doIt();
	}
	
	MGlobal::displayInfo(MString("proxy paint extracted ") + numSelected + " transforms in " + m_groupCount + " groups");
}

void proxyPaintContext::erectSelected()
{
	if(!m_pViz)
		return;
		
	unsigned numSelected = m_pViz->getNumActiveBoxes();
	
	if(numSelected < 1) return;
	
	MDagModifier mod;
	MStatus stat;
	Vector3F worldUp(0.f, 1.f, 0.f);	
	for(unsigned i = 0; i < numSelected; i++)
	{
		MMatrix mat = m_pViz->getActiveBox(i);
		
		Vector3F vx(mat(0, 0), mat(0, 1), mat(0, 2));
		Vector3F vy(mat(1, 0), mat(1, 1), mat(1, 2));
		Vector3F vz(mat(2, 0), mat(2, 1), mat(2, 2));
		
		float sx = vx.length();
		float sy = vy.length();
		float sz = vz.length();
		
		vx.y = 0.f;
		vx.normalize();
		
		vz = vx.cross(worldUp);
		vz.normalize();
		
		vx *= sx;
		vy = Vector3F(0.f, sy, 0.f);
		vz *= sz;
		
		mat(0, 0) = vx.x;
		mat(0, 1) = vx.y;
		mat(0, 2) = vx.z;
		mat(1, 0) = vy.x;
		mat(1, 1) = vy.y;
		mat(1, 2) = vy.z;
		mat(2, 0) = vz.x;
		mat(2, 1) = vz.y;
		mat(2, 2) = vz.z;
		
		m_pViz->setActiveBox(i, mat);
	}
	
	MGlobal::displayInfo(MString("proxy paint erected ") + numSelected + " transforms");
}

void proxyPaintContext::sendCullSurface()
{
	if(!validateSelection()) return;
			
	if(goCollide && m_cullSelection == 1) {
		m_pViz->setCullMesh(fcollide.dagPath());
	}
	else {
		MDagPath nouse;
		m_pViz->setCullMesh(nouse);
	}
}

void proxyPaintContext::snap()
{}

void proxyPaintContext::erase()
{
    if(!m_pViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	m_pViz->erase(fromNear, fromFar, m_brushWeight);
	view.refresh( true );	
}

void proxyPaintContext::startProcessSelect()
{
	if(!m_pViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld (start_x, start_y, fromNear, fromFar );
	
	m_pViz->selectPlant(fromNear, fromFar, MGlobal::kReplaceList);
}

void proxyPaintContext::processSelect()
{
	if(!m_pViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	m_pViz->selectPlant(fromNear, fromFar, m_listAdjustment);
}

char proxyPaintContext::validateViz(const MSelectionList &sels)
{
    MStatus stat;
    MItSelectionList iter(sels, MFn::kPluginLocatorNode, &stat );
    MObject vizobj;
    iter.getDependNode(vizobj);
    if(vizobj != MObject::kNullObj)
	{
        MFnDependencyNode fviz(vizobj);
		m_pViz = (ProxyViz*)fviz.userNode();
	}
    
    if(!m_pViz)
        return 0;
    
    return 1;
}

char proxyPaintContext::validateCollide(const MSelectionList &sels)
{
	goCollide = 0;
	
    if(fcollide.setObject(m_activeMeshPath) == MS::kSuccess)
	{
		goCollide = 1;    
		return 1;
	}
		
	MStatus stat;
	MItSelectionList meshIter(sels, MFn::kMesh, &stat);
		
	MDagPath mo;
	meshIter.getDagPath( mo );
	if(fcollide.setObject(mo) == MS::kSuccess) 
	{
		goCollide = 1;    
		return 1;
	}
	
    return 0;
}

void proxyPaintContext::setGrowOption(ProxyViz::GrowOption & opt)
{
	opt.m_upDirection = Vector3F::YAxis;
	opt.m_alongNormal = m_growAlongNormal > 0;
	opt.m_minScale = m_min_scale;
	opt.m_maxScale = m_max_scale;
	opt.m_rotateNoise = m_rotation_noise;
	opt.m_marginSize = 1.f;
	opt.m_plantId = 0;
	opt.m_multiGrow = m_multiCreate;
}

void proxyPaintContext::setWriteCache(MString filename)
{
	MGlobal::displayInfo(MString("proxyPaint tries to write to cache ") + filename);
	if(!getSelectedViz())
		return;
	m_pViz->pressToSave();
}

void proxyPaintContext::setReadCache(MString filename)
{
	MGlobal::displayInfo(MString("proxyPaint tries to read from cache ") + filename);
	if(!getSelectedViz())
		return;
	m_pViz->pressToLoad();
}

void proxyPaintContext::cleanup()
{
	MGlobal::displayInfo("proxyPaint set to reset");
	if(!getSelectedViz())
		return;
	m_pViz->removeAllPlants();	
}

void proxyPaintContext::finishGrow()
{
	if(!m_pViz) return;
	m_pViz->finishGrow();
}

char proxyPaintContext::getSelectedViz()
{
	MSelectionList slist;
	MGlobal::getActiveSelectionList( slist );
	if(!validateViz(slist)) {
		MGlobal::displayWarning("No proxyViz selected");
		return 0;
	}
	return 1;
}

proxyPaintContextCmd::proxyPaintContextCmd() {}

MPxContext* proxyPaintContextCmd::makeObj()
{
	fContext = new proxyPaintContext();
	return fContext;
}

void* proxyPaintContextCmd::creator()
{
	return new proxyPaintContextCmd;
}

MStatus proxyPaintContextCmd::doEditFlags()
{
	MStatus status = MS::kSuccess;
	
	MArgParser argData = parser();
	
	if (argData.isFlagSet(kOptFlag)) 
	{
		unsigned mode;
		status = argData.getFlagArgument(kOptFlag, 0, mode);
		if (!status) {
			status.perror("mode flag parsing failed.");
			return status;
		}
		fContext->setOperation(mode);
	}
	
	if (argData.isFlagSet(kNsegFlag)) 
	{
		unsigned nseg;
		status = argData.getFlagArgument(kNsegFlag, 0, nseg);
		if (!status) {
			status.perror("nseg flag parsing failed.");
			return status;
		}
		fContext->setNSegment(nseg);
	}
	
	if (argData.isFlagSet(kLsegFlag)) 
	{
		double lseg;
		status = argData.getFlagArgument(kLsegFlag, 0, lseg);
		if (!status) {
			status.perror("lseg flag parsing failed.");
			return status;
		}
		fContext->setBrushRadius(lseg);
	}
	
	if (argData.isFlagSet(kWeightFlag)) 
	{
		double wei;
		status = argData.getFlagArgument(kWeightFlag, 0, wei);
		if (!status) {
			status.perror("lseg flag parsing failed.");
			return status;
		}
		fContext->setBrushWeight(wei);
	}
	
	if (argData.isFlagSet(kMinFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kMinFlag, 0, noi);
		if (!status) {
			status.perror("scale min flag parsing failed.");
			return status;
		}
		fContext->setScaleMin(noi);
	}
	
	if (argData.isFlagSet(kMaxFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kMaxFlag, 0, noi);
		if (!status) {
			status.perror("scale max flag parsing failed.");
			return status;
		}
		fContext->setScaleMax(noi);
	}
	
	if (argData.isFlagSet(kRotateNoiseFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kRotateNoiseFlag, 0, noi);
		if (!status) {
			status.perror("rotate noise flag parsing failed.");
			return status;
		}
		fContext->setRotationNoise(noi);
	}
	
	if (argData.isFlagSet(kNormalFlag)) 
	{
		unsigned aln;
		status = argData.getFlagArgument(kNormalFlag, 0, aln);
		if (!status) {
			status.perror("normal flag parsing failed.");
			return status;
		}
		fContext->setGrowAlongNormal(aln);
	}
	
	if (argData.isFlagSet(kWriteCacheFlag)) 
	{
		MString ch;
		status = argData.getFlagArgument(kWriteCacheFlag, 0, ch);
		if (!status) {
			status.perror("cache out flag parsing failed.");
			return status;
		}
		fContext->setWriteCache(ch);
	}
	
	if (argData.isFlagSet(kReadCacheFlag)) 
	{
		MString ch;
		status = argData.getFlagArgument(kReadCacheFlag, 0, ch);
		if (!status) {
			status.perror("cache in flag parsing failed.");
			return status;
		}
		fContext->setReadCache(ch);
	}
	
	if (argData.isFlagSet(kCullSelectionFlag)) {
		unsigned cus;
		status = argData.getFlagArgument(kCullSelectionFlag, 0, cus);
		if (!status) {
			status.perror("cull selection flag parsing failed.");
			return status;
		}
		fContext->setCullSelection(cus);
	}
	
	if (argData.isFlagSet(kMultiCreateFlag)) {
		unsigned mcr;
		status = argData.getFlagArgument(kMultiCreateFlag, 0, mcr);
		if (!status) {
			status.perror("multi create flag parsing failed.");
			return status;
		}
		fContext->setMultiCreate(mcr);
	}
	
	if (argData.isFlagSet(kInstanceGroupCountFlag)) {
		unsigned igc;
		status = argData.getFlagArgument(kInstanceGroupCountFlag, 0, igc);
		if (!status) {
			status.perror("instance group count flag parsing failed.");
			return status;
		}
		fContext->setInstanceGroupCount(igc);
	}

	return MS::kSuccess;
}

MStatus proxyPaintContextCmd::doQueryFlags()
{
	MArgParser argData = parser();
	
	if (argData.isFlagSet(kOptFlag)) {
		setResult((int)fContext->getOperation());
	}
	
	if (argData.isFlagSet(kNsegFlag)) {
		setResult((int)fContext->getNSegment());
	}
	
	if (argData.isFlagSet(kLsegFlag)) {
		setResult((float)fContext->getBrushRadius());
	}
	
	if (argData.isFlagSet(kWeightFlag)) {
		setResult((float)fContext->getBrushWeight());
	}
	
	if (argData.isFlagSet(kMinFlag)) {
		setResult((float)fContext->getScaleMin());
	}
	
	if (argData.isFlagSet(kMaxFlag)) {
		setResult((float)fContext->getScaleMax());
	}
	
	if (argData.isFlagSet(kRotateNoiseFlag)) {
		setResult((float)fContext->getRotationNoise());
	}
	
	if (argData.isFlagSet(kNormalFlag)) {
		setResult((int)fContext->getGrowAlongNormal());
	}
	
	if (argData.isFlagSet(kCullSelectionFlag)) {
		setResult((int)fContext->getCullSelection());
	}
	
	if (argData.isFlagSet(kMultiCreateFlag)) {
		setResult((int)fContext->getMultiCreate());
	}
	
	if (argData.isFlagSet(kInstanceGroupCountFlag)) {
		setResult((int)fContext->getInstanceGroupCount());
	}
	
	return MS::kSuccess;
}

MStatus proxyPaintContextCmd::appendSyntax()
{
	MSyntax mySyntax = syntax();
	
	MStatus stat;
	stat = mySyntax.addFlag(kOptFlag, kOptFlagLong, MSyntax::kUnsigned);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add option arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNsegFlag, kNsegFlagLong, MSyntax::kUnsigned);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add numseg arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kLsegFlag, kLsegFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add radius arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kWeightFlag, kWeightFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add weight arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kMinFlag, kMinFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add min arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kMaxFlag, kMaxFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add max arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kNormalFlag, kNormalFlagLong, MSyntax::kUnsigned);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add normal arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kWriteCacheFlag, kWriteCacheFlagLong, MSyntax::kString);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add cache out arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kReadCacheFlag, kReadCacheFlagLong, MSyntax::kString);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add cache in arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kRotateNoiseFlag, kRotateNoiseFlagLong, MSyntax::kDouble);
	if(!stat)
	{
		MGlobal::displayInfo("failed to add rotate noise arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kCullSelectionFlag, kCullSelectionFlagLong, MSyntax::kUnsigned);
	if(!stat) {
		MGlobal::displayInfo("failed to add cull selection arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kMultiCreateFlag, kMultiCreateFlagLong, MSyntax::kUnsigned);
	if(!stat) {
		MGlobal::displayInfo("failed to add multi create arg");
		return MS::kFailure;
	}
	
	stat = mySyntax.addFlag(kInstanceGroupCountFlag, kInstanceGroupCountFlagLong, MSyntax::kUnsigned);
	if(!stat) {
		MGlobal::displayInfo("failed to add instance group count arg");
		return MS::kFailure;
	}
	
	return MS::kSuccess;
}
//:~