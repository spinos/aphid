#include "boxPaintTool.h"
#include <maya/MFnCamera.h>
#include <maya/MPointArray.h>
#include <maya/MDagModifier.h>
#include <maya/MToolsInfo.h>
#include <ASearchHelper.h>

const char helpString[] =
			"Select a proxy viz to paint on";

ProxyViz * proxyPaintContext::PtrViz = NULL;

proxyPaintContext::proxyPaintContext():mOpt(opSelect),
m_brushWeight(.66f),m_min_scale(1.f),m_max_scale(1.f),m_rotation_noise(0.f),
m_growAlongNormal(0),
m_createMargin(0.1f), 
m_multiCreate(0),
m_extractGroupCount(1),
m_plantType(0)
{
	setTitleString ( "proxyPaint Tool" );

	// Tell the context which XPM to use so the tool can properly
	// be a candidate for the 6th position on the mini-bar.
	setImage("proxyPaintTool.xpm", MPxContext::kImage1 );
	attachSceneCallbacks();
}

proxyPaintContext::~proxyPaintContext()
{ detachSceneCallbacks(); }

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

	MDagPath camera;
	view.getCamera( camera );
	
	MFnCamera fnCamera( camera );
	
	clipNear = fnCamera.nearClippingPlane();
	clipFar = fnCamera.farClippingPlane();

	validateSelection();
    
    if(event.isModifierShift()) m_currentOpt = opResizeBrush;
    else m_currentOpt = mOpt;

	if(m_currentOpt == opSelect) startProcessSelect();
	if(m_currentOpt == opSelectGround) startSelectGround();
	
	return MS::kSuccess;		
}


MStatus proxyPaintContext::doDrag( MEvent & event )
{
	event.getPosition( last_x, last_y );

	switch (m_currentOpt)
	{
		case opCreate :
			grow();
			break;
		case opErase :
			erase();
			break;
		case opSelect : 
			processSelect();
			break;
		case opResize :
			resize();
			break;
		case opMove :
			move();
			break;
		case opRotateY :
			rotateAroundAxis(1);
			break;
		case opRotateZ :
			rotateAroundAxis(2);
			break;
		case opRotateX :
			rotateAroundAxis(0);
			break;
		case opSelectGround :
			selectGround();
			break;
		case opReplace :
			replace();
			break;
        case opResizeBrush :
            scaleBrush();
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
	
	if(!PtrViz) return MS::kSuccess;
	if(m_currentOpt==opErase) PtrViz->finishErase();
	if(m_currentOpt==opSelect) AHelper::Info<unsigned>("n active plants", PtrViz->numActivePlants() );
	if(m_currentOpt==opSelectGround) AHelper::Info<unsigned>("n active faces", PtrViz->numActiveGroundFaces() );
	
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

void proxyPaintContext::setOperation(short val)
{
	if(val == opClean) {
		cleanup();
		return;
	}
	
	if(val==opFlood) {
		flood();
		return;
	}
	
	if(val ==opExtract) {
		extractSelected();
		return;
	}
	
    std::string opstr("unknown");
	mOpt = opUnknown;
	switch (val)
	{
		case opCreate:
			opstr="create";
            mOpt = opCreate;
			break;
		case opErase: 
			opstr="erase";
            mOpt = opErase;
			break;
		case opSelect: 
			opstr="select";
            mOpt =opSelect;
			break;
		case opResize:
			opstr="scale";
            mOpt = opResize;
			break;
		case opMove:
			opstr="move";
            mOpt = opMove;
			break;
		case opRotateY:
			opstr="rotate y";
            mOpt = opRotateY;
			break;
		case opRotateZ:
			opstr="rotate z";
            mOpt = opRotateZ;
			break;
		case opRotateX:
			opstr="rotate x";
            mOpt = opRotateX;
			break;
		case opResizeBrush:
			opstr="resize brush";
            mOpt = opResizeBrush;
			break;
		case opSelectGround:
			opstr="ground faces";
            mOpt = opSelectGround;
			break;
		case opReplace:
			opstr="replace";
            mOpt = opReplace;
			break;
		default:
			;
	}
	AHelper::Info<std::string>("proxyPaintTool set operation mode", opstr);
    MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getOperation() const
{ return mOpt; }

void proxyPaintContext::setBrushRadius(float val)
{
    if(PtrViz)
        PtrViz->setSelectionRadius(val);
        
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getBrushRadius() const
{
    if(PtrViz)
        return PtrViz->selectionRadius();
        
	return 8.f;
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
{ return m_multiCreate; }

void proxyPaintContext::setInstanceGroupCount(unsigned val)
{
	MGlobal::displayInfo(MString("proxyPaint will extract transforms into ") + val + " groups");
	m_extractGroupCount = val;
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getInstanceGroupCount() const
{ return m_extractGroupCount; }

void proxyPaintContext::scaleBrush()
{
    if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
		
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	PtrViz->adjustBrushSize(fromNear, fromFar, mag);
}

void proxyPaintContext::resize()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
		
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	
    PtrViz->setNoiseWeight(m_rotation_noise);
	PtrViz->adjustSize(fromNear, fromFar, mag);
}

void proxyPaintContext::move()
{
	if(!PtrViz) return;
		
	PtrViz->setNoiseWeight(m_rotation_noise);
	PtrViz->adjustPosition(start_x, start_y, last_x, last_y,  clipNear, clipFar);
}

void proxyPaintContext::rotateAroundAxis(short axis)
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
		
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	
    PtrViz->setNoiseWeight(m_rotation_noise);
	PtrViz->adjustRotation(fromNear, fromFar, mag, axis);
}

void proxyPaintContext::moveAlongAxis(short axis)
{}

void proxyPaintContext::selectGround()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	PtrViz->setNoiseWeight(m_rotation_noise);
	PtrViz->selectGround(fromNear, fromFar, m_listAdjustment);
}

void proxyPaintContext::startSelectGround()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld (start_x, start_y, fromNear, fromFar );
	
	PtrViz->selectGround(fromNear, fromFar, MGlobal::kReplaceList);
}

void proxyPaintContext::smoothSelected()
{}

void proxyPaintContext::grow()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	ProxyViz::GrowOption opt;
	setGrowOption(opt);
	PtrViz->grow(fromNear, fromFar, opt);
}

char proxyPaintContext::validateSelection()
{
	MSelectionList slist;
 	MGlobal::getActiveSelectionList( slist );
	if(!validateViz(slist))
	    MGlobal::displayWarning("No proxyViz selected");
			
    if(!PtrViz) return 0;

	PtrViz->setSelectionRadius(getBrushRadius() );
	return 1;
}

void proxyPaintContext::flood()
{
	if(!PtrViz) return;
	ProxyViz::GrowOption opt;
	setGrowOption(opt);
	PtrViz->flood(opt);
}

void proxyPaintContext::extractSelected()
{
	if(!PtrViz) return;
	PtrViz->extractActive(m_extractGroupCount);
}

void proxyPaintContext::erectSelected()
{
	if(!PtrViz) return;
	PtrViz->erectActive();
}

void proxyPaintContext::snap()
{}

void proxyPaintContext::erase()
{
    if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	ProxyViz::GrowOption opt;
	setGrowOption(opt);
	PtrViz->erase(fromNear, fromFar, opt);
	view.refresh( true );	
}

void proxyPaintContext::startProcessSelect()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld (start_x, start_y, fromNear, fromFar );
	
	PtrViz->selectPlant(fromNear, fromFar, MGlobal::kReplaceList);
}

void proxyPaintContext::processSelect()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	PtrViz->selectPlant(fromNear, fromFar, m_listAdjustment);
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
		if(fviz.typeName() != "proxyViz") {
			PtrViz = NULL;
			return 0;
		}
		PtrViz = (ProxyViz*)fviz.userNode();
	}
    
    if(!PtrViz)
        return 0;
    
    return 1;
}

void proxyPaintContext::setGrowOption(ProxyViz::GrowOption & opt)
{
	opt.m_upDirection = Vector3F::YAxis;
	opt.m_alongNormal = m_growAlongNormal > 0;
	opt.m_minScale = m_min_scale;
	opt.m_maxScale = m_max_scale;
	opt.m_rotateNoise = m_rotation_noise;
	opt.m_marginSize = 1.f;
	opt.m_plantId = m_plantType;
	opt.m_multiGrow = m_multiCreate;
	opt.m_marginSize = m_createMargin;
	opt.m_strength = m_brushWeight;
}

void proxyPaintContext::setWriteCache(MString filename)
{
	MGlobal::displayInfo(MString("proxyPaint tries to write to cache ") + filename);
	if(!getSelectedViz())
		return;
	PtrViz->pressToSave();
}

void proxyPaintContext::setReadCache(MString filename)
{
	MGlobal::displayInfo(MString("proxyPaint tries to read from cache ") + filename);
	if(!getSelectedViz())
		return;
	PtrViz->pressToLoad();
}

void proxyPaintContext::cleanup()
{
	MGlobal::displayInfo("proxyPaint set to reset");
	if(!getSelectedViz())
		return;
	PtrViz->removeAllPlants();	
}

void proxyPaintContext::finishGrow()
{
	if(!PtrViz) return;
	PtrViz->finishGrow();
}

void proxyPaintContext::replace()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	ProxyViz::GrowOption opt;
	setGrowOption(opt);
	PtrViz->replacePlant(fromNear, fromFar, opt);
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

void proxyPaintContext::setCreateMargin(float x)
{ m_createMargin = x; }

const float & proxyPaintContext::createMargin()
{ return m_createMargin; }

void proxyPaintContext::setPlantType(int x)
{ 
	AHelper::Info<int>(" proxyPaintContext select plant", x);
	m_plantType = x; 
}

const int & proxyPaintContext::plantType() const
{ return m_plantType; }

void proxyPaintContext::attachSceneCallbacks()
{
	fBeforeNewCB  = MSceneMessage::addCallback(MSceneMessage::kBeforeNew,  releaseCallback, this);
	fBeforeOpenCB  = MSceneMessage::addCallback(MSceneMessage::kBeforeOpen,  releaseCallback, this);
}

void proxyPaintContext::detachSceneCallbacks()
{
	if (fBeforeNewCB)
		MMessage::removeCallback(fBeforeNewCB);
	if (fBeforeOpenCB)
		MMessage::removeCallback(fBeforeOpenCB);
	fBeforeNewCB = 0;
	fBeforeOpenCB = 0;
}

void proxyPaintContext::releaseCallback(void* clientData)
{ PtrViz = NULL; }
	
//:~