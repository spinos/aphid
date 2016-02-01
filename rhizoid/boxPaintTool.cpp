#include "boxPaintTool.h"
#include <maya/MFnCamera.h>
#include <maya/MPointArray.h>
#include <maya/MDagModifier.h>
#include "../core/TriangleRaster.h"
#include <ASearchHelper.h>

proxyPaintContext::proxyPaintContext():mOpt(999),m_numSeg(5),m_brushRadius(8.f),m_brushWeight(.66f),m_min_scale(1.f),m_max_scale(1.f),m_rotation_noise(0.f),m_pViz(0),
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
		//snap();
		return;
	}
	
	if(val == 102) {
		extractSelected();
		return;
	}
	
	if(val == 103) {
		//erectSelected();
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
			
    if(!m_pViz) return 0;

	m_pViz->setSelectionRadius(getBrushRadius() );
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
	if(!m_pViz) return;
	m_pViz->extractActive(m_groupCount);
}

void proxyPaintContext::erectSelected()
{
	if(!m_pViz) return;
	m_pViz->erectActive();
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
//:~