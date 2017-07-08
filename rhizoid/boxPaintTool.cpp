#include "boxPaintTool.h"
#include <maya/MFnCamera.h>
#include <maya/MPointArray.h>
#include <maya/MDagModifier.h>
#include <maya/MToolsInfo.h>
#include <maya/MFnParticleSystem.h>
#include <mama/AHelper.h>
#include <mama/ASearchHelper.h>
#include "ProxyVizNode.h"

using namespace aphid;

const char helpString[] = "Select a proxy viz to paint on";

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
	if ( event.isModifierControl() ) {
		m_listAdjustment = MGlobal::kRemoveFromList;
	}
	
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
    
    if(event.isModifierShift()) {
        m_currentOpt = opResizeBrush;
    } else {
        m_currentOpt = mOpt;
    }
    
    switch (m_currentOpt) {
		case opResizeBrush:
            startResizeBrush();
            break;
        case opSelectGround:
            startSelectGround();
            break;
        case opSelect:
        case opSelectByType:
            startProcessSelect();
            break;
		case opBundleResize:
            startResize();
            break;
        case opBundleRotate:
            startRotate();
            break;
        case opBundleTranslate:
            startTranslate();
            break;
		case opRaise:
		case opDepress:
            startEditOffset();
            break;
        default:
            ;
    }

	return MS::kSuccess;		
}


MStatus proxyPaintContext::doDrag( MEvent & event )
{
	if(!PtrViz) {
		return MS::kSuccess;
	}
	
	event.getPosition( last_x, last_y );

	switch (m_currentOpt) {
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
			resize(false);
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
			processSelectGround();
			break;
		case opReplace :
			replace();
			break;
        case opResizeBrush :
            processResizeBrush();
			break;
		case opRaise :
            raiseOffset();
            break;
		case opDepress :
            depressOffset();
            break;
		case opBundleResize :
            processResize();
            break;
		case opBundleRotate :
            processRotate();
            break;
        case opBundleTranslate :
            processTranslate();
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
	
	switch (m_currentOpt) {
        case opSelectGround :
            PtrViz->finishGroundSelection();
            break;
		case opSelect :
		case opSelectByType :
			PtrViz->finishPlantSelection();
		case opBundleResize:
            PtrViz->finishResize();
            break;
		case opBundleRotate :
            PtrViz->finishRotate();
            break;
        case opBundleTranslate :
            PtrViz->finishTranslate();
            break;
		default:
		    ;
	}
	
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
	bool toReturn = false;
	switch (val) {
		case opReshuffle:
			reshuffleSamples();
			toReturn = true;
		break;
		case opClearOffset:
			clearOffset();
			toReturn = true;
		break;
		case opRandMove:
			moveRandomly();
			toReturn = true;
		break;
		case opRandRotate:
			rotateRandomly();
			toReturn = true;
		break;
		case opRandResize:
			resizeSelectedRandomly();
			toReturn = true;
		break;
		case opInjectParticle:
			injectSelectedParticle();
			toReturn = true;
			break;
		case opInjectTransform:
			injectSelectedTransform();
			toReturn = true;
		break;
		case opDiscardSampleSelection:
			discardSampleSelection();
			toReturn = true;
		break;
		case opDiscardPlantSelection:
			discardPlantSelection();
			toReturn = true;
		break;
		case opClean:
			clearAllPlants();
			toReturn = true;
		break;
		case opFlood:
			flood();
			toReturn = true;
		break;
		case opExtract:
			extractSelected();
			toReturn = true;
		break;
		case opCleanByType:
			clearByType();
			toReturn = true;
		break;
		case opErect:
			erect();
			toReturn = true;
		break;
		default:
		;
	}
	
	if(toReturn) {
		return;
	}
	
	ModifyForest::ManipulateMode khand = ModifyForest::manNone;
	
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
			opstr="resize";
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
			opstr="select ground samples";
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
		case opBundleResize:
			opstr="bundle resize";
            mOpt = opBundleResize;
			khand = ModifyForest::manScaling;
			break;
		case opBundleRotate:
			opstr="bundle rotate";
            mOpt = opBundleRotate;
            khand = ModifyForest::manRotate;
			break;
        case opBundleTranslate:
			opstr="bundle move";
            mOpt = opBundleTranslate;
            khand = ModifyForest::manTranslate;
			break;
		default:
			;
	}
	AHelper::Info<std::string>("proxyPaintTool set operation mode", opstr);
	setManipulator(khand);
    MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getOperation() const
{ return mOpt; }

void proxyPaintContext::setBrushRadius(float val)
{
	validateSelection();
    if(PtrViz) {
        PtrViz->processBrushRadius(val);
	}
        
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getBrushRadius() const
{
    if(PtrViz) {
        return PtrViz->selectionRadius();
	}
        
	return 8.f;
}

float proxyPaintContext::zenithNoise() const
{ return m_growOpt.m_zenithNoise; }

void proxyPaintContext::setZenithNoise(float x)
{ 
	m_growOpt.m_zenithNoise = x; 
	MToolsInfo::setDirtyFlag(*this);
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
	if(val == 1) {
		MGlobal::displayInfo("proxyPaint enable grow along face normal");
	} else {
		MGlobal::displayInfo("proxyPaint disable grow along face normal");
	}
	m_growOpt.m_alongNormal = val;
/// reset up anyway
	m_growOpt.m_upDirection = Vector3F::YAxis;
	MToolsInfo::setDirtyFlag(*this);
}

unsigned proxyPaintContext::getGrowAlongNormal() const
{ return m_growOpt.m_alongNormal; }

void proxyPaintContext::setMultiCreate(unsigned val)
{
	if(val > 0) { 
		MGlobal::displayInfo("proxyPaint enable multiple create");
	} else {
		MGlobal::displayInfo("proxyPaint disable multiple create");
	}
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

void proxyPaintContext::startResizeBrush()
{
    if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( start_x, start_y, fromNear, fromFar );
	
	PtrViz->beginAdjustBrushSize(fromNear, fromFar, m_growOpt);
}

void proxyPaintContext::processResizeBrush()
{
    if(!PtrViz) return;
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	PtrViz->adjustBrushSize(mag);
}

void proxyPaintContext::resize(bool isBundled)
{
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
		
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	
    PtrViz->setNoiseWeight(m_growOpt.m_rotateNoise);
	PtrViz->adjustSize(fromNear, fromFar, mag, isBundled);
}

void proxyPaintContext::move()
{
	if(!PtrViz) return;
	if(rejectSmallDragDistance() )
        return;
		
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

void proxyPaintContext::processSelectGround()
{   
	if(!PtrViz) {
		std::cout<<"\n selectGround has no PtrViz";
		return;
	}
	if(rejectSmallDragDistance() ) {
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
	if(!PtrViz) {
		return;
	}
	
	MPoint fromNear, fromFar;
	view.viewToWorld (start_x, start_y, fromNear, fromFar );
	
	PtrViz->selectGround(fromNear, fromFar, m_listAdjustment);
}

void proxyPaintContext::smoothSelected()
{}

void proxyPaintContext::grow()
{
    if(rejectSmallDragDistance() )
        return;
        
	if(!PtrViz) return;
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	PtrViz->grow(fromNear, fromFar, m_growOpt);
}

void proxyPaintContext::flood()
{
	if(!PtrViz) return;
/// no radius limit
	m_growOpt.m_radius = -1.f;
	PtrViz->processFlood(m_growOpt);
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
	m_growOpt.m_upDirection = Vector3F::YAxis;
	PtrViz->rightUp(m_growOpt);
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
    if(rejectSmallDragDistance() ) return;
        
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
	if(rejectSmallDragDistance() ) return;
        
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	PtrViz->selectPlantByType(fromNear, fromFar, -1, m_listAdjustment);
}

void proxyPaintContext::processSelectByType()
{
    if(!PtrViz) return;
    if(rejectSmallDragDistance() ) return;
        
	MPoint fromNear, fromFar;
	view.viewToWorld ( last_x, last_y, fromNear, fromFar );
	
	PtrViz->selectPlantByType(fromNear, fromFar, m_growOpt.m_plantId, m_listAdjustment);
}

void proxyPaintContext::setWriteCache(MString filename)
{
	MGlobal::displayInfo(MString("proxyPaint tries to write to cache ") + filename);
	if(!validateSelection())
		return;
	PtrViz->pressToSave();
}

void proxyPaintContext::setReadCache(MString filename)
{
	MGlobal::displayInfo(MString("proxyPaint tries to read from cache ") + filename);
	if(!validateSelection())
		return;
	PtrViz->pressToLoad();
}

void proxyPaintContext::clearAllPlants()
{
	MGlobal::displayInfo("proxyPaint set to clear all plants");
	if(!validateSelection())
		return;
	PtrViz->processClearAllPlants();	
}

void proxyPaintContext::clearBySelections()
{
    if(!validateSelection())
		return;
	MGlobal::displayInfo("proxyPaint set to clear all selected");
	PtrViz->processRemoveActivePlants();	
}

void proxyPaintContext::clearByType()
{
	if(!validateSelection())
		return;
	if(PtrViz->numActivePlants() > 0 ) {
	    clearBySelections();
	    return;
	}
	PtrViz->processRemoveTypedPlants(m_growOpt);	
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
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	m_growOpt.setStrokeMagnitude(mag);
	PtrViz->offsetAlongNormal(m_growOpt);
}

void proxyPaintContext::depressOffset()
{
	float mag = last_x - start_x - last_y + start_y;
	mag /= 48;
	m_growOpt.setStrokeMagnitude(mag * -1.f);
	PtrViz->offsetAlongNormal(m_growOpt);
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

void proxyPaintContext::discardSampleSelection()
{
    if(!PtrViz) return;
    PtrViz->processDeselectSamples();
}

void proxyPaintContext::discardPlantSelection()
{
    if(!PtrViz) return;
    PtrViz->processDeselectPlants();
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
{ 
	PtrViz = NULL; 
	ObjViz = MObject::kNullObj;
}

void proxyPaintContext::injectSelectedParticle()
{
	if(!PtrViz) {
        MGlobal::displayWarning("proxyPaintContext has no active viz");
		return;
    }
    
    MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintContext wrong selection, select particle system(s) to inject");
		return;
	}
	
	MStatus stat;
    MItSelectionList parIter(sels, MFn::kParticle, &stat);
	if(!stat) {
		MGlobal::displayWarning("proxyPaintContext no particle system selected, nothing to inject");
		return;
	}
	
	m_growOpt.m_isInjectingParticle = true;
	for(;!parIter.isDone();parIter.next() ) {
		MObject parNode;
		stat = parIter.getDependNode(parNode);
		if(!stat) {
			MGlobal::displayWarning("proxyPaintContext no particle system selected, nothing to inject");
			continue;
		}

		MFnParticleSystem parFn(parNode, &stat);
		if(!stat) {
			AHelper::Info<MString>("not a particle system", MFnDependencyNode(parNode).name() );
			continue;
		}

		MVectorArray pos;
		parFn.position(pos);

		const unsigned np = pos.length();
			
		std::vector<Matrix44F> ms;
		Matrix44F wmf;
		for(unsigned i=0;i < np;++i ) {
			wmf.setTranslation(Vector3F(pos[i].x, pos[i].y, pos[i].z) );
			ms.push_back(wmf);
		}
		
		AHelper::Info<unsigned>("proxyPaintContext inject n particle", np );
		PtrViz->injectPlants(ms, m_growOpt);
		ms.clear();
	}
	AHelper::Info<int>("proxyPaintContext created n plant", PtrViz->numActivePlants() );
    
}

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
        wmd = AHelper::GetWorldTransformMatrix(transPath);
        AHelper::ConvertToMatrix44F(wmf, wmd);
        ms.push_back(wmf);
	}
    
    AHelper::Info<int>("proxyPaintContext inject n transform", ms.size() );
    m_growOpt.m_isInjectingParticle = false;
	PtrViz->injectPlants(ms, m_growOpt);
	AHelper::Info<int>("proxyPaintContext created n plant", PtrViz->numActivePlants() );
    
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
	processFilterNoise(); 
}

void proxyPaintContext::setNoiseLacunarity(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise lacunarity", x);
	m_growOpt.m_noiseLacunarity = x; 
	processFilterNoise();
}

void proxyPaintContext::setNoiseOctave(int x)
{ 
	AHelper::Info<int>(" proxyPaintContext set noise octace", x);
	m_growOpt.m_noiseOctave = x;
	processFilterNoise(); 
}

void proxyPaintContext::setNoiseLevel(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise level", x);
	m_growOpt.m_noiseLevel = x; 
	processFilterNoise();
}

void proxyPaintContext::setNoiseGain(float x)
{ 
	AHelper::Info<float>(" proxyPaintContext set noise gain", x);
	m_growOpt.m_noiseGain = x; 
	processFilterNoise();
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

void proxyPaintContext::setNoiseOrigin(float x, float y, float z)
{
	AHelper::Info<Vector3F>(" proxyPaintContext set noise origin", Vector3F(x, y, z) );
	m_growOpt.m_noiseOrigin.set(x, y, z);
	processFilterNoise();
}

/// limit frequency of action
bool proxyPaintContext::rejectSmallDragDistance(int d) const
{
	return (Absolute<int>(last_x - start_x) 
        + Absolute<int>(last_y - start_y)) < d;
}

void proxyPaintContext::setManipulator(int x)
{
    if(!PtrViz) {
		return;
	}
	ModifyForest::ManipulateMode md = ModifyForest::manNone;
	switch(x) {
	case ModifyForest::manRotate :
	    md = ModifyForest::manRotate;
	    break;
	    case ModifyForest::manTranslate :
	    md = ModifyForest::manTranslate;
	    break;
	    case ModifyForest::manScaling :
	    md = ModifyForest::manScaling;
	    break;
	}
	PtrViz->processManipulatMode(md, m_growOpt);
}

void proxyPaintContext::startRotate()
{
	PtrViz->startRotate(getIncidentAt(start_x, start_y) );
}

void proxyPaintContext::processRotate()
{
	PtrViz->processRotate(getIncidentAt(last_x, last_y) );
}

void proxyPaintContext::startTranslate()
{
    PtrViz->startTranslate(getIncidentAt(start_x, start_y) );
}
	
void proxyPaintContext::processTranslate()
{
	PtrViz->processTranslate(getIncidentAt(last_x, last_y) );
}

void proxyPaintContext::startResize()
{
    PtrViz->startResize(getIncidentAt(start_x, start_y) );
}
	
void proxyPaintContext::processResize()
{
	PtrViz->processResize(getIncidentAt(last_x, last_y) );
}

Ray proxyPaintContext::getIncidentAt(int x, int y)
{
    MPoint fromNear, fromFar;
	view.viewToWorld ( x, y, fromNear, fromFar );
	Vector3F a(fromNear.x, fromNear.y, fromNear.z);
	Vector3F b(fromFar.x, fromFar.y, fromFar.z);
	return Ray(a, b);
}

void proxyPaintContext::setImageSamplerName(MString filename)
{
	if(filename.length() < 5) {
		MGlobal::displayInfo("proxyPaintContext remove image sampler");
		m_growOpt.closeImage();
		
	} else {
		bool stat = m_growOpt.openImage(filename.asChar() );
		if(stat) {
			AHelper::Info<MString>("proxyPaintContext opened image sampler", filename);
		} else {
			AHelper::Info<MString>("proxyPaintContext cannot open image", filename);
		}
	}
	if(PtrViz) {
	     PtrViz->processFilterImage(m_growOpt);   
	}
	MToolsInfo::setDirtyFlag(*this);
}

MString proxyPaintContext::imageSamplerName() const
{
	if(!m_growOpt.hasSampler() ) {
		return MString("unknown");
	}
	return MString(m_growOpt.imageName().c_str() );
}

void proxyPaintContext::reshuffleSamples()
{
	validateSelection();
	if(PtrViz) {
		PtrViz->processReshuffle();
	}
}

void proxyPaintContext::viewDependentSelectSamples()
{
	validateSelection();
	if(PtrViz) {
		PtrViz->processViewDependentSelectSamples();
	}
}

void proxyPaintContext::setFilterPortion(float x)
{
	if(PtrViz) {
		PtrViz->processFilterPortion(x);
	}
}
	
float proxyPaintContext::filterPortion() const
{
	if(!PtrViz) {
		return 1.f;
	}
	return PtrViz->filterPortion();
}

void proxyPaintContext::processFilterNoise()
{
	validateSelection();
	if(PtrViz) {
		PtrViz->processFilterNoise(m_growOpt);
	}
}

int proxyPaintContext::numVisibleSamples()
{
	int c = 0;
	validateSelection();
	if(PtrViz) {
		c = PtrViz->numVisibleSamples();
	}
	return c;
}

void proxyPaintContext::setShowVizGrid(int x)
{
	validateSelection();
	if(PtrViz) {
		PtrViz->processSetShowGrid(x);
	}
}

int proxyPaintContext::getShowVizGrid()
{
	int r = 0;
	validateSelection();
	if(PtrViz) {
		r = PtrViz->getShowGrid();
	}
	return r;
}

void proxyPaintContext::setEditVizGround(int x)
{
	validateSelection();
	if(PtrViz) {
		PtrViz->processSetFastGround(x);
	}
}
	
int proxyPaintContext::getEditVizGround()
{
	int r = -1;
	validateSelection();
	if(PtrViz) {
		r = PtrViz->getFastGround();
	}
	return r;
}

void proxyPaintContext::setBrushFalloff(float x)
{
    m_growOpt.setbrushFalloff(x);
    AHelper::Info<float>("proxyPaintContext set brush fall off", m_growOpt.m_brushFalloff);
    
    validateSelection();
    if(PtrViz) {
        PtrViz->processBrushFalloff(x);
	}
	
	MToolsInfo::setDirtyFlag(*this);
}

float proxyPaintContext::getBrushFalloff()
{
	return m_growOpt.m_brushFalloff;
}

void proxyPaintContext::setShowVoxelThreshold(float x)
{
	processShowVoxelThreshold(x);
	
	MToolsInfo::setDirtyFlag(*this);
}

void proxyPaintContext::startEditOffset()
{
	if(!PtrViz) {
		return;
	}
	PtrViz->processStartEditOffset(getIncidentAt(start_x, start_y) );
}
//:~