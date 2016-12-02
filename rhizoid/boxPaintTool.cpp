#include "boxPaintTool.h"
#include <maya/MFnCamera.h>
#include <maya/MPointArray.h>
#include <maya/MDagModifier.h>
#include <maya/MToolsInfo.h>
#include <ASearchHelper.h>

const char helpString[] =
			"Select a proxy viz to paint on";

ProxyViz * proxyPaintContext::PtrViz = NULL;

proxyPaintContext::proxyPaintContext() : mOpt(opSelect),
m_extractGroupCount(1)
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
	
	if(clipFar > 1e7f) {
		AHelper::Info<float>("[WARNING] truncate camera far clipping plane to ", 1e7f);
		clipFar = 1e7f;
	}
    
    if(event.isModifierShift()) m_currentOpt = opResizeBrush;
    else m_currentOpt = mOpt;

	if(m_currentOpt == opSelect || m_currentOpt == opSelectByType) startProcessSelect();
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
        case opSelectByType : 
			processSelectByType();
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
        case opRotateToDir :
            rotateByStroke();
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
		case opRaise :
            raiseOffset();
            break;
		case opDepress :
            depressOffset();
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
	if(m_currentOpt==opErase ||
		m_currentOpt==opMove ) PtrViz->finishErase();
	if(m_currentOpt==opSelect || m_currentOpt==opSelectByType) AHelper::Info<unsigned>("n active plants", PtrViz->numActivePlants() );
	if(m_currentOpt==opSelectGround)
	    PtrViz->finishGroundSelection(m_growOpt);
	
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
	if(val == opClearOffset) {
		clearOffset();
		return;
	}
	
	if(val == opRandMove) {
		moveRandomly();
		return;
	}
	
	if(val == opRandRotate) {
		rotateRandomly();
		return;
	}
	
	if(val == opRandResize) {
		resizeSelectedRandomly();
		return;
	}
	
    if(val == opInjectTransform) {
		injectSelectedTransform();
		return;
	}
    
    if(val == opDiscardFaceSelection) {
		discardFaceSelection();
		return;
	}
    
    if(val == opDiscardPlantSelection) {
		discardPlantSelection();
		return;
	}
    
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
	
	if(val == opCleanByType) {
		clearByType();
		return;
	}
	
	if(val == opErect) {
		erect();
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
        case opSelectByType:
			opstr="select by type";
            mOpt = opSelectByType;
			break;
        case opRotateToDir:
			opstr="rotate to direction";
            mOpt = opRotateToDir;
			break;
		case opRaise:
			opstr="raise offset";
            mOpt = opRaise;
			break;
		case opDepress:
			opstr="depress offset";
            mOpt = opDepress;
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
	m_growOpt.m_minScale = val;
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getScaleMin() const
{
	return m_growOpt.m_minScale;
}

void proxyPaintContext::setScaleMax(float val)
{
	m_growOpt.m_maxScale = val;
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getScaleMax() const
{
	return m_growOpt.m_maxScale;
}

void proxyPaintContext::setRotationNoise(float val)
{
	m_growOpt.m_rotateNoise = val;
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getRotationNoise() const
{
	return m_growOpt.m_rotateNoise;
}

void proxyPaintContext::setBrushWeight(float val)
{
	m_growOpt.m_strength = val;
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getBrushWeight() const
{
	return m_growOpt.m_strength;
}

void proxyPaintContext::setGrowAlongNormal(unsigned val)
{
	if(val == 1) 
		MGlobal::displayInfo("proxyPaint enable grow along face normal");
	else
		MGlobal::displayInfo("proxyPaint disable grow along face normal");
	m_growOpt.m_alongNormal = val;
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getGrowAlongNormal() const
{ return m_growOpt.m_alongNormal; }

void proxyPaintContext::setMultiCreate(unsigned val)
{
	if(val == 1) 
		MGlobal::displayInfo("proxyPaint enable multiple create");
	else
		MGlobal::displayInfo("proxyPaint disable multiple create");
	m_growOpt.m_multiGrow = val;
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getMultiCreate() const
{ return m_growOpt.m_multiGrow; }

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
	
    PtrViz->setNoiseWeight(m_growOpt.m_rotateNoise);
	PtrViz->adjustSize(fromNear, fromFar, mag);
}

void proxyPaintContext::move()
{
	if(!PtrViz) return;
		
	PtrViz->setNoiseWeight(m_growOpt.m_rotateNoise);
	PtrViz->adjustPosition(start_x, start_y, last_x, last_y,  clipNear, clipFar);
}

void proxyPaintContext::rotateAroundAxis(short axis)
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
		
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	
    PtrViz->setNoiseWeight(m_growOpt.m_rotateNoise);
	PtrViz->adjustRotation(fromNear, fromFar, mag, axis);
}

void proxyPaintContext::moveAlongAxis(short axis)
{}

void proxyPaintContext::selectGround()
{
	if(!PtrViz) {
		std::cout<<"\n selectGround has no PtrViz";
		return;
	}
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	PtrViz->setNoiseWeight(m_growOpt.m_rotateNoise);
	PtrViz->selectGround(fromNear, fromFar, m_listAdjustment);
}

void proxyPaintContext::startSelectGround()
{
	validateSelection();
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld (start_x, start_y, fromNear, fromFar );
	
	PtrViz->selectGround(fromNear, fromFar, MGlobal::kReplaceList);
}

void proxyPaintContext::smoothSelected()
{}

void proxyPaintContext::grow()
{
/// limit frequency of action
    if(Absolute<int>(last_x - start_x) 
        + Absolute<int>(last_y - start_y) < 3)
        return;
        
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	PtrViz->grow(fromNear, fromFar, m_growOpt);
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
/// no radius limit
	m_growOpt.m_radius = -1.f;
	PtrViz->flood(m_growOpt);
}

void proxyPaintContext::extractSelected()
{
	if(!PtrViz) return;
	PtrViz->extractActive(m_extractGroupCount);
}

void proxyPaintContext::erect()
{
	if(!PtrViz) return;
	MGlobal::displayInfo("proxyPaint right up");
	PtrViz->erectActive();
}

void proxyPaintContext::rotateByStroke()
{
	if(!PtrViz) return;
	PtrViz->setNoiseWeight(m_growOpt.m_rotateNoise);
	PtrViz->rotateToDirection(start_x, start_y, last_x, last_y,  clipNear, clipFar);
}

void proxyPaintContext::snap()
{}

void proxyPaintContext::erase()
{
    if(!PtrViz) return;
    if(Absolute<int>(last_x - start_x) 
        + Absolute<int>(last_y - start_y) < 2)
        return;
        
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	PtrViz->erase(fromNear, fromFar, m_growOpt);
	view.refresh( true );	
}

void proxyPaintContext::startProcessSelect()
{
	validateSelection();
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld (start_x, start_y, fromNear, fromFar );
	
	PtrViz->selectPlantByType(fromNear, fromFar, -1, MGlobal::kReplaceList);
}

void proxyPaintContext::processSelect()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	PtrViz->selectPlantByType(fromNear, fromFar, -1, m_listAdjustment);
}

void proxyPaintContext::processSelectByType()
{
    if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	PtrViz->selectPlantByType(fromNear, fromFar, m_growOpt.m_plantId, m_listAdjustment);
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

void proxyPaintContext::clearByType()
{
	MGlobal::displayInfo("proxyPaint set to clear by type");
	if(!getSelectedViz())
		return;
	AHelper::Info<int>("active plant type", m_growOpt.m_plantId);
	PtrViz->removeTypedPlants(m_growOpt.m_plantId);	
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
	PtrViz->replacePlant(fromNear, fromFar, m_growOpt);
}

void proxyPaintContext::raiseOffset()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	m_growOpt.setStrokeMagnitude(mag);
	PtrViz->offsetAlongNormal(fromNear, fromFar, m_growOpt);
}

void proxyPaintContext::depressOffset()
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	m_growOpt.setStrokeMagnitude(mag * -1.f);
	PtrViz->offsetAlongNormal(fromNear, fromFar, m_growOpt);
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

void proxyPaintContext::setMinCreateMargin(float x)
{ m_growOpt.m_minMarginSize = x > .1f ? x : .1f; }

const float & proxyPaintContext::minCreateMargin()
{ return m_growOpt.m_minMarginSize; }

void proxyPaintContext::setMaxCreateMargin(float x)
{ m_growOpt.m_maxMarginSize = x > .1f ? x : .1f; }

const float & proxyPaintContext::maxCreateMargin()
{ return m_growOpt.m_maxMarginSize; }

void proxyPaintContext::setPlantType(int x)
{ 
	AHelper::Info<int>(" proxyPaintContext select plant", x);
	m_growOpt.m_plantId = x; 
}

const int & proxyPaintContext::plantType() const
{ return m_growOpt.m_plantId; }

void proxyPaintContext::discardFaceSelection()
{
    if(!PtrViz) return;
    PtrViz->deselectFaces();
}

void proxyPaintContext::discardPlantSelection()
{
    if(!PtrViz) return;
    PtrViz->deselectPlants();
}

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

void proxyPaintContext::injectSelectedTransform()
{
    if(!PtrViz) {
        MGlobal::displayWarning("proxyPaintContext has no active viz");
		return;
    }
    
    MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintContext wrong selection, select transform(s) to inject");
		return;
	}
    
    MStatus stat;
    MItSelectionList transIter(sels, MFn::kTransform, &stat);
	if(!stat) {
		MGlobal::displayWarning("proxyPaintContext no transform selected, nothing to inject");
		return;
	}
    
    std::vector<Matrix44F> ms;
    Matrix44F wmf;
    MMatrix wmd;
    for(;!transIter.isDone(); transIter.next() ) {
		MDagPath transPath;
		transIter.getDagPath(transPath);
        wmd = aphid::AHelper::GetWorldTransformMatrix(transPath);
        AHelper::ConvertToMatrix44F(wmf, wmd);
        ms.push_back(wmf);
	}
    
    AHelper::Info<int>("proxyPaintContext inject n transform", ms.size() );
    PtrViz->injectPlants(ms, m_growOpt);
}

void proxyPaintContext::resizeSelectedRandomly()
{
	if(!PtrViz) return;
	MGlobal::displayInfo("proxyPaintContext scale randomly");
	PtrViz->scalePlant(m_growOpt);
}

void proxyPaintContext::moveRandomly()
{
	if(!PtrViz) return;
	AHelper::Info<float>("proxyPaintContext move randomly by max margin ", m_growOpt.m_maxMarginSize);
	PtrViz->movePlant(m_growOpt);
}

void proxyPaintContext::rotateRandomly()
{
	if(!PtrViz) return;
	AHelper::Info<float>("proxyPaintContext move randomly by rotate noise ", m_growOpt.m_rotateNoise);
	PtrViz->rotatePlant(m_growOpt);
}

void proxyPaintContext::clearOffset()
{
	if(!PtrViz) return;
	AHelper::Info<Vector3F >("proxyPaintContext reset offset ", Vector3F::Zero);
	PtrViz->clearPlantOffset(m_growOpt);
}

void proxyPaintContext::setStickToGround(bool x)
{ 
	AHelper::Info<bool>(" proxyPaintContext set stick to ground", x);
	m_growOpt.m_stickToGround = x; 
}

const bool & proxyPaintContext::stickToGround() const
{ return m_growOpt.m_stickToGround; }

void proxyPaintContext::selectViz()
{ validateSelection(); }

const float & proxyPaintContext::noiseFrequency() const
{ return m_growOpt.m_noiseFrequency; }

const float & proxyPaintContext::noiseLacunarity() const
{ return m_growOpt.m_noiseLacunarity; }

const int & proxyPaintContext::noiseOctave() const
{ return m_growOpt.m_noiseOctave; }

const float & proxyPaintContext::noiseLevel() const
{ return m_growOpt.m_noiseLevel; }

const float & proxyPaintContext::noiseGain() const
{ return m_growOpt.m_noiseGain; }

const float & proxyPaintContext::noiseOriginX() const
{ return m_growOpt.m_noiseOrigin.x; }

const float & proxyPaintContext::noiseOriginY() const
{ return m_growOpt.m_noiseOrigin.y; }

const float & proxyPaintContext::noiseOriginZ() const
{ return m_growOpt.m_noiseOrigin.z; }

void proxyPaintContext::setNoiseFrequency(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise frequency", x);
	m_growOpt.m_noiseFrequency = x; 
}

void proxyPaintContext::setNoiseLacunarity(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise lacunarity", x);
	m_growOpt.m_noiseLacunarity = x; 
}

void proxyPaintContext::setNoiseOctave(int x)
{ 
	AHelper::Info<int>(" proxyPaintContext set noise octace", x);
	m_growOpt.m_noiseOctave = x; 
}

void proxyPaintContext::setNoiseLevel(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise level", x);
	m_growOpt.m_noiseLevel = x; 
}

void proxyPaintContext::setNoiseGain(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise gain", x);
	m_growOpt.m_noiseGain = x; 
}

void proxyPaintContext::setNoiseOriginX(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise origin x", x);
	m_growOpt.m_noiseOrigin.x = x; 
}

void proxyPaintContext::setNoiseOriginY(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise origin y", x);
	m_growOpt.m_noiseOrigin.y = x; 
}

void proxyPaintContext::setNoiseOriginZ(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise origin z", x);
	m_growOpt.m_noiseOrigin.z = x; 
}
//:~