#include <Vector2F.h>
#include "proxyVizNode.h"
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MVectorArray.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MEulerRotation.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MIntArray.h>
#include <maya/MPointArray.h>
#include <maya/MFnMeshData.h>
#include <PseudoNoise.h>
#include <EnvVar.h>
#include <AHelper.h>
#include <fstream> 

MTypeId ProxyViz::id( 0x95a19e );
MObject ProxyViz::abboxminx;
MObject ProxyViz::abboxminy;
MObject ProxyViz::abboxminz;
MObject ProxyViz::abboxmaxx;
MObject ProxyViz::abboxmaxy;
MObject ProxyViz::abboxmaxz;
MObject ProxyViz::outPositionPP;
MObject ProxyViz::outScalePP;
MObject ProxyViz::outRotationPP;
MObject ProxyViz::outValue;
MObject ProxyViz::acachename;
MObject ProxyViz::adumpname;
MObject ProxyViz::acameraspace;
MObject ProxyViz::ahapeture;
MObject ProxyViz::avapeture;
MObject ProxyViz::afocallength;
MObject ProxyViz::alodgatehigh;
MObject ProxyViz::alodgatelow;
MObject ProxyViz::axmultiplier;
MObject ProxyViz::aymultiplier;
MObject ProxyViz::azmultiplier;
MObject ProxyViz::agroupcount;
MObject ProxyViz::astarttime;
MObject ProxyViz::ainstanceId;
MObject ProxyViz::aenablecull;
MObject ProxyViz::ainmesh;
MObject ProxyViz::astandinNames;
MObject ProxyViz::aconvertPercentage;
MObject ProxyViz::agroundMesh;
MObject ProxyViz::aplantTransformCache;
MObject ProxyViz::aplantIdCache;
MObject ProxyViz::aplantTriangleIdCache;
MObject ProxyViz::aplantTriangleCoordCache;

ProxyViz::ProxyViz() : _firstLoad(1), fHasView(0), fVisibleTag(0), fCuller(0),
m_toSetGrid(true), m_hasCamera(false)
{ attachSceneCallbacks(); }

ProxyViz::~ProxyViz() 
{
    if(fVisibleTag) delete[] fVisibleTag;
	detachSceneCallbacks();
}

MStatus ProxyViz::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
		
		updateWorldSpace();
		
		MStatus status;

		BoundingBox * defb = defBoxP();
		defb->setMin(block.inputValue(abboxminx).asFloat(), 0);
		defb->setMin(block.inputValue(abboxminy).asFloat(), 1);
		defb->setMin(block.inputValue(abboxminz).asFloat(), 2);
		defb->setMax(block.inputValue(abboxmaxx).asFloat(), 0);
		defb->setMax(block.inputValue(abboxmaxy).asFloat(), 1);
		defb->setMax(block.inputValue(abboxmaxz).asFloat(), 2);
		
		if(m_toSetGrid) {
			m_toSetGrid = false;
			resetGrid(defb->distance(0) * 20.f);
		}
		
		if(_firstLoad) {
/// internal cache has the priority
			if(!loadInternal(block) ) {
				MString filename =  block.inputValue( acachename ).asString();
				if(filename != "") {
					loadExternal(replaceEnvVar(filename).c_str());
				}
			}
			_firstLoad = 0;
		}
        
		MArrayDataHandle groundArray = block.inputArrayValue(agroundMesh );
        updateGround(groundArray );
		moveWithGround();
		
		MMatrix cameraInv;
		MDataHandle cameradata = block.inputValue(acameraspace, &status);
        if(status) cameraInv = cameradata.asMatrix();
		
		fDisplayMesh = block.inputValue( ainmesh ).asMesh();
		
		double h_apeture = block.inputValue(ahapeture).asDouble();
		double v_apeture = block.inputValue(avapeture).asDouble();
		double fl = block.inputValue(afocallength).asDouble();
		double h_fov = h_apeture * 0.5 / ( fl * 0.03937 );
		float gate_high = block.inputValue(alodgatehigh).asFloat();
		float gate_low = block.inputValue(alodgatelow).asFloat();
		float start_time = block.inputValue(astarttime).asFloat();
		int groupCount = block.inputValue(agroupcount).asInt();
		int groupId = block.inputValue(ainstanceId).asInt();
		int frustumCull = block.inputValue(aenablecull).asInt();
		const double percentage = block.inputValue(aconvertPercentage).asDouble();
		m_materializePercentage = percentage;
		MString bakename =  block.inputValue( adumpname ).asString();
		
/// particle output
		unsigned num_box = _spaces.length();
		_details.setLength(num_box);
		_randNums.setLength(num_box);
		
		if(fVisibleTag) delete[] fVisibleTag;
		fVisibleTag = new char[num_box];
		
		if(start_time == 1.f) {
			MGlobal::displayInfo(MString("proxy viz initialize lod keys"));
			PseudoNoise pnoise;
			for(unsigned i =1; i < num_box; i++) {
				_details[i] = -1.f;
				_randNums[i] = pnoise.rint1(i + 2397 * i, num_box * 4);
				fVisibleTag[i] = 1;
			}
			_activeIndices.clear();
		}

		const Vector3F vdetail = defb->getMax() - defb->getMin();
		const float detail = vdetail.length();
		float aspectRatio = v_apeture / h_apeture;
		calculateLOD(cameraInv, h_fov, aspectRatio, detail, frustumCull);
		
		MPlug plgParticlePos(thisMObject(), outPositionPP);
		if(!plgParticlePos.isConnected()) {
			block.setClean(plug);
            return MS::kSuccess;
		}
		
		MDataHandle hdata = block.inputValue(outPositionPP, &status);
        MFnVectorArrayData farray(hdata.data(), &status);
        if(!status) {
            MGlobal::displayInfo("proxy viz is not properly connected to a particle system");
			block.setClean(plug);
            return MS::kSuccess;
        }
    
        MDataHandle scaledata = block.inputValue(outScalePP, &status);
        MFnVectorArrayData scalearray(scaledata.data(), &status);
        if(!status) {
            MGlobal::displayInfo("proxy viz is not properly connected to a particle system");
			block.setClean(plug);
            return MS::kSuccess;
        }
		
		MDataHandle rotatedata = block.inputValue(outRotationPP, &status);
        MFnVectorArrayData rotatearray(rotatedata.data(), &status);
        if(!status) {
            MGlobal::displayInfo("proxy viz is not properly connected to a particle system");
			block.setClean(plug);
            return MS::kSuccess;
        }
		
		MVectorArray outPosArray = farray.array();	
        MVectorArray outScaleArray = scalearray.array();
		MVectorArray outRotateArray = rotatearray.array();
		
		if( outPosArray.length() < 1) {
			block.setClean(plug);
			return MS::kSuccess;
		}
		
		MGlobal::displayInfo(MString("proxy viz computes particle attributes"));
        
        outPosArray.clear();
        outScaleArray.clear();
		outRotateArray.clear();
		
		MGlobal::displayInfo(MString("proxy viz boxes count: ") + num_box);
		
		if(gate_high >= 1.f)
			gate_high = 10e8;

        for(unsigned i =1; i < num_box; i++) {
			if(frustumCull > 0) {
				if(_details[i] >= gate_high || _details[i] < gate_low) 
					continue;
			}
				
			if(groupCount > 1) {
				int grp = _randNums[i] % groupCount;
				if(grp != groupId)
					continue;
			}
			
			if(percentage < 1.0) {
			    double dart = ((double)(rand()%497))/497.0;
			    if(dart > percentage) continue;
			}
			
			fVisibleTag[i] = 0;
                
            const MMatrix space = worldizeSpace(_spaces[i]);
            const MVector pos(space(3,0), space(3,1), space(3,2));
			const MVector vx(space(0,0), space(0,1), space(0,2));
            const double sz = vx.length();
			 
            outPosArray.append(pos);
                
            
            const MVector scale(sz, sz, sz);
            outScaleArray.append(scale);
			
			MEulerRotation eula;
			eula = space; 
			outRotateArray.append(eula.asVector());

        }
		
		if(outPosArray.length() < 1) {
			outPosArray.append(MVector(0,0,0));
			outScaleArray.append(MVector(1,1,1));
			outRotateArray.append(MVector(0,0,0));
		}
		
		MGlobal::displayInfo(MString("proxy viz gen ") + outPosArray.length() + " instances for group " + groupId);
		
		if(bakename != "") {
		    MGlobal::displayInfo(MString("proxy viz bake result to ") + bakename);
			bakePass(replaceEnvVar(bakename).c_str(), outPosArray, outScaleArray, outRotateArray);
		}

        float result = outPosArray.length();

		MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set( result );
		block.setClean(plug);
    }

	return MS::kSuccess;
}

void ProxyViz::draw( M3dView & view, const MDagPath & path, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{ 	
	updateWorldSpace();
	MObject thisNode = thisMObject();
	
	MPlug mutxplug( thisNode, axmultiplier);
	setScaleMuliplier(mutxplug.asFloat(), 0);
	
	MPlug mutyplug( thisNode, aymultiplier);
	setScaleMuliplier(mutyplug.asFloat(), 1);
	
	MPlug mutzplug( thisNode, azmultiplier);
	setScaleMuliplier(mutzplug.asFloat(), 2);	
	
	_viewport = view;
	fHasView = 1;

	view.beginGL();
	
	if(m_hasCamera) vizViewFrustum(thisNode);
	
	if(!fCuller)
		fCuller = new DepthCut;
	
	if(!fCuller->isDiagnosed()) {
#ifdef WIN32
        MGlobal::displayInfo("init glext on win32");
		gExtensionInit();
#endif		
		std::string log;
		fCuller->diagnose(log);
		MGlobal::displayInfo(MString("glsl diagnose log: ") + log.c_str());
	}
	
	double mm[16];
	if(fCuller->isDiagnosed()) {
		if(!fCuller->hasFBO()) {
			std::string log;
			fCuller->initializeFBO(log);
			MGlobal::displayInfo(log.c_str());
		}
		matrix_as_array(_worldInverseSpace, mm);
		fCuller->setLocalSpace(mm);
		fCuller->frameBufferBegin();
		fCuller->drawFrameBuffer();
		fCuller->frameBufferEnd();
		//fCuller->showFrameBuffer();
	}
	
	draw_a_box();
	drawGridBounding();
	// drawGrid();

	if ( style == M3dView::kFlatShaded || 
		    style == M3dView::kGouraudShaded ) {		
		drawPlants();
	}
	
	drawWiredPlants();
	drawActivePlants();
	drawGround();
	view.endGL();
}

bool ProxyViz::isBounded() const
{ return true; }

MBoundingBox ProxyViz::boundingBox() const
{   
	BoundingBox bbox = defBox();
	if(numPlants() > 0) bbox = gridBoundingBox();
	
	MPoint corner1(bbox.m_data[0], bbox.m_data[1], bbox.m_data[2]);
	MPoint corner2(bbox.m_data[3], bbox.m_data[4], bbox.m_data[5]);

	return MBoundingBox( corner1, corner2 );
}

void* ProxyViz::creator()
{
	return new ProxyViz();
}

MStatus ProxyViz::initialize()
{ 
	MFnNumericAttribute numFn;
	MStatus			 stat;
	
	alodgatehigh = numFn.create( "lodGateMax", "ldmx", MFnNumericData::kFloat, 1.f);
	numFn.setKeyable(true);
	numFn.setStorable(true);
	numFn.setMin(0.001f);
	numFn.setMax(2.f);
	addAttribute(alodgatehigh);
	
	alodgatelow = numFn.create( "lodGateMin", "ldmin", MFnNumericData::kFloat, 0.f);
	numFn.setKeyable(true);
	numFn.setStorable(true);
	numFn.setMin(0.f);
	numFn.setMax(0.999f);
	addAttribute(alodgatelow);

	abboxminx = numFn.create( "bBoxMinX", "bbmnx", MFnNumericData::kFloat, -1.f );
	numFn.setKeyable(true);
	numFn.setStorable(true);
	addAttribute(abboxminx);
	
	abboxminy = numFn.create( "bBoxMinY", "bbmny", MFnNumericData::kFloat, -1.f );
	numFn.setKeyable(true);
	numFn.setStorable(true);
	addAttribute(abboxminy);
	
	abboxminz = numFn.create( "bBoxMinZ", "bbmnz", MFnNumericData::kFloat, -1.f );
	numFn.setKeyable(true);
	numFn.setStorable(true);
	addAttribute(abboxminz);
	
	abboxmaxx = numFn.create( "bBoxMaxX", "bbmxx", MFnNumericData::kFloat, 1.f );
	numFn.setKeyable(true);
	numFn.setStorable(true);
	addAttribute(abboxmaxx);
	
	abboxmaxy = numFn.create( "bBoxMaxY", "bbmxy", MFnNumericData::kFloat, 1.f);
	numFn.setKeyable(true);
	numFn.setStorable(true);
	addAttribute(abboxmaxy);
	
	abboxmaxz = numFn.create( "bBoxMaxZ", "bbmxz", MFnNumericData::kFloat, 1.f);
	numFn.setKeyable(true);
	numFn.setStorable(true);
	addAttribute(abboxmaxz);
	
	axmultiplier = numFn.create( "visualMultiplierX", "vmx", MFnNumericData::kFloat, 1.f);
	numFn.setKeyable(true);
	numFn.setStorable(true);
	numFn.setMin(0.001f);
	addAttribute(axmultiplier);	
	aymultiplier = numFn.create( "visualMultiplierY", "vmy", MFnNumericData::kFloat, 1.f);
	numFn.setKeyable(true);
	numFn.setStorable(true);
	numFn.setMin(0.001f);
	addAttribute(aymultiplier);
	
	azmultiplier = numFn.create( "visualMultiplierZ", "vmz", MFnNumericData::kFloat, 1.f);
	numFn.setKeyable(true);
	numFn.setStorable(true);
	numFn.setMin(0.001f);
	addAttribute(azmultiplier);
	
	agroupcount = numFn.create( "numberInstances", "nis", MFnNumericData::kInt, 1);
	numFn.setKeyable(false);
	numFn.setStorable(true);
	numFn.setMin(1);
	addAttribute(agroupcount);
	
	astarttime = numFn.create( "startTime", "stt", MFnNumericData::kFloat, 1.f);
	numFn.setKeyable(false);
	numFn.setStorable(true);
	addAttribute(astarttime);
	
	ainstanceId = numFn.create( "instanceId", "iis", MFnNumericData::kInt, 0);
	numFn.setKeyable(false);
	numFn.setStorable(true);
	numFn.setMin(0);
	addAttribute(ainstanceId);
	
	aenablecull = numFn.create( "enableCull", "ecl", MFnNumericData::kInt, 1);
	numFn.setKeyable(false);
	numFn.setStorable(true);
	numFn.setMin(0);
	numFn.setMax(1);
	addAttribute(aenablecull);

	MFnTypedAttribute typedAttrFn;
	MVectorArray defaultVectArray;
	MFnVectorArrayData vectArrayDataFn;
	vectArrayDataFn.create( defaultVectArray );
	outPositionPP = typedAttrFn.create( "outPosition",
											"opos",
											MFnData::kVectorArray,
											vectArrayDataFn.object(),
											&stat );
												
												if(!stat) MGlobal::displayWarning("failed create pospp");

	typedAttrFn.setStorable(false);
	if(addAttribute( outPositionPP ) != MS::kSuccess) MGlobal::displayWarning("failed add pospp");
	
	outScalePP = typedAttrFn.create( "outScale",
											"oscl",
											MFnData::kVectorArray,
											vectArrayDataFn.object(),
											&stat );
												
												if(!stat) MGlobal::displayWarning("failed create sclpp");

        typedAttrFn.setStorable(false);
        if(addAttribute(outScalePP) != MS::kSuccess) MGlobal::displayWarning("failed add sclpp");
		
	outRotationPP = typedAttrFn.create( "outRotation",
											"orot",
											MFnData::kVectorArray,
											vectArrayDataFn.object(),
											&stat );
												
												if(!stat) MGlobal::displayWarning("failed create rotpp");

        typedAttrFn.setStorable(false);
        if(addAttribute(outRotationPP) != MS::kSuccess) MGlobal::displayWarning("failed add rotpp");
		

        
    outValue = numFn.create( "outValue", "ov", MFnNumericData::kFloat );
	numFn.setStorable(false);
	numFn.setWritable(false);
	addAttribute(outValue);
	
	MFnTypedAttribute   stringAttr;
	acachename = stringAttr.create( "cachePath", "cp", MFnData::kString );
 	stringAttr.setStorable(true);
	addAttribute( acachename );
	
	adumpname = stringAttr.create( "bakePath", "bkp", MFnData::kString );
 	stringAttr.setStorable(true);
 	addAttribute(adumpname);
	
	astandinNames = stringAttr.create( "standinNames", "sdn", MFnData::kString );
 	stringAttr.setStorable(true);
	stringAttr.setArray(true);
	addAttribute(astandinNames);
	
	MFnMatrixAttribute matAttr;
	acameraspace = matAttr.create( "cameraSpace", "cspc", MFnMatrixAttribute::kDouble );
 	matAttr.setStorable(false);
	matAttr.setWritable(true);
	matAttr.setConnectable(true);
	addAttribute(acameraspace);
	
	ahapeture = numFn.create( "horizontalFilmAperture", "hfa", MFnNumericData::kDouble, 1.0 );
	numFn.setStorable(false);
	numFn.setConnectable(true);
	addAttribute( ahapeture );
	
	avapeture = numFn.create( "verticalFilmAperture", "vfa", MFnNumericData::kDouble, 1.0 );
	numFn.setStorable(false);
	numFn.setConnectable(true);
	addAttribute( avapeture );
	
	afocallength = numFn.create( "focalLength", "fl", MFnNumericData::kDouble );
	numFn.setStorable(false);
	numFn.setConnectable(true);
	addAttribute( afocallength );
	
	ainmesh = typedAttrFn.create("displayMesh", "dspm", MFnMeshData::kMesh);
	typedAttrFn.setStorable(false);
	typedAttrFn.setWritable(true);
	typedAttrFn.setConnectable(true);
	addAttribute( ainmesh );
	
	aconvertPercentage = numFn.create( "convertPercentage", "cvp", MFnNumericData::kDouble );
	numFn.setStorable(false);
	numFn.setConnectable(true);
	numFn.setDefault(1.0);
	numFn.setMax(1.0);
	numFn.setMin(0.01);
	addAttribute(aconvertPercentage);
    
    agroundMesh = typedAttrFn.create("groundMesh", "grdm", MFnMeshData::kMesh);
	typedAttrFn.setStorable(false);
	typedAttrFn.setWritable(true);
	typedAttrFn.setConnectable(true);
    typedAttrFn.setArray(true);
    typedAttrFn.setDisconnectBehavior(MFnAttribute::kDelete);
	addAttribute( agroundMesh );
	attributeAffects(agroundMesh, outValue);
	
	MPointArray defaultPntArray;
	MFnPointArrayData pntArrayDataFn;
	pntArrayDataFn.create( defaultPntArray );
	aplantTransformCache = typedAttrFn.create( "transformCachePlant",
											"tmcpl",
											MFnData::kPointArray,
											pntArrayDataFn.object(),
											&stat );
    typedAttrFn.setStorable(true);
	addAttribute(aplantTransformCache);
	
	MIntArray defaultIntArray;
	MFnIntArrayData intArrayDataFn;
	intArrayDataFn.create( defaultIntArray );
	aplantIdCache = typedAttrFn.create( "idCachePlant",
											"idcpl",
											MFnData::kIntArray,
											intArrayDataFn.object(),
											&stat );
    typedAttrFn.setStorable(true);
	addAttribute(aplantIdCache);
	
	aplantTriangleIdCache = typedAttrFn.create( "triCachePlant",
											"trcpl",
											MFnData::kIntArray,
											intArrayDataFn.object(),
											&stat );
    typedAttrFn.setStorable(true);
	addAttribute(aplantTriangleIdCache);
	
	aplantTriangleCoordCache = typedAttrFn.create( "coordCachePlant",
											"crcpl",
											MFnData::kVectorArray,
											vectArrayDataFn.object(),
											&stat );
    typedAttrFn.setStorable(true);
	addAttribute(aplantTriangleCoordCache);
	
	// attributeAffects(acameraspace, outValue);
	attributeAffects(abboxminx, outValue);
	attributeAffects(abboxmaxx, outValue);
	attributeAffects(abboxminy, outValue);
	attributeAffects(abboxmaxy, outValue);
	attributeAffects(abboxminz, outValue);
	attributeAffects(abboxmaxz, outValue);
	attributeAffects(outPositionPP, outValue);
	attributeAffects(alodgatehigh, outValue);
	attributeAffects(alodgatelow, outValue);
	attributeAffects(astarttime, outValue);
	attributeAffects(ainstanceId, outValue);
	attributeAffects(aenablecull, outValue);
	attributeAffects(ainmesh, outValue);
	attributeAffects(aconvertPercentage, outValue);

	return MS::kSuccess;
}

void ProxyViz::attachSceneCallbacks()
{
	fBeforeSaveCB  = MSceneMessage::addCallback(MSceneMessage::kBeforeSave,  releaseCallback, this);
}

void ProxyViz::detachSceneCallbacks()
{
	if (fBeforeSaveCB)
		MMessage::removeCallback(fBeforeSaveCB);

	fBeforeSaveCB = 0;
}

void ProxyViz::releaseCallback(void* clientData)
{
	ProxyViz *pThis = (ProxyViz*) clientData;
	pThis->saveInternal();
}

void ProxyViz::saveInternal()
{
	AHelper::Info<MString>("prxnode save internal", MFnDependencyNode(thisMObject()).name() );
	updateNumPlants();
	const unsigned n = numPlants();
	AHelper::Info<unsigned>("num plants", n );
	if(n<1) return;
	
	MPointArray plantTms;
	MIntArray plantIds;
	MIntArray plantTris;
	MVectorArray plantCoords;
	
	savePlants(plantTms, plantIds, plantTris, plantCoords);
	
	MFnPointArrayData tmFn;
	MObject otm = tmFn.create(plantTms);
	MPlug tmPlug(thisMObject(), aplantTransformCache);
	tmPlug.setValue(otm);
	
	MFnIntArrayData idFn;
	MObject oid = idFn.create(plantIds);
	MPlug idPlug(thisMObject(), aplantIdCache);
	idPlug.setValue(oid);
	
	MFnIntArrayData triFn;
	MObject otri = idFn.create(plantTris);
	MPlug triPlug(thisMObject(), aplantTriangleIdCache);
	triPlug.setValue(otri);
	
	MFnVectorArrayData crdFn;
	MObject ocrd = crdFn.create(plantCoords);
	MPlug crdPlug(thisMObject(), aplantTriangleCoordCache);
	crdPlug.setValue(ocrd);
}

bool ProxyViz::loadInternal(MDataBlock& block)
{
	MDataHandle tmH = block.inputValue(aplantTransformCache);
	MFnPointArrayData tmFn(tmH.data());
	MPointArray plantTms = tmFn.array();
	if(plantTms.length() < 1) return false;
	
	MDataHandle idH = block.inputValue(aplantIdCache);
	MFnIntArrayData idFn(idH.data());
	MIntArray plantIds = idFn.array();
	if(plantIds.length() < 1) return false;
	
	MDataHandle triH = block.inputValue(aplantTriangleIdCache);
	MFnIntArrayData triFn(triH.data());
	MIntArray plantTris = triFn.array();
	if(plantTris.length() < 1) return false;
	
	MDataHandle crdH = block.inputValue(aplantTriangleCoordCache);
	MFnVectorArrayData crdFn(crdH.data());
	MVectorArray plantCoords = crdFn.array();
	if(plantCoords.length() < 1) return false;
		
	return loadPlants(plantTms, plantIds, plantTris, plantCoords);
}

char ProxyViz::isBoxInView(const MPoint &pos, float threshold, short xmin, short ymin, short xmax, short ymax)
{
	int portW = _viewport.portWidth();
	int portH = _viewport.portHeight();
	MMatrix modelViewMatrix;
	_viewport.modelViewMatrix ( modelViewMatrix );
	const MPoint pcam = pos * modelViewMatrix;
	short x, y;
	_viewport.worldToView (pos, x, y);
		
	if(x > xmin && x < xmax && y > ymin && y < ymax) {
		if(!fCuller)
			return 1;

		if(fCuller->isDiagnosed()) {
			
			const float depth = -pcam.z;
			if(fCuller->isCulled(depth, x, y, portW, portH, threshold))
				return 0;
		}
		return 1;
	}
	return 0;
}

void ProxyViz::adjustSize(short x, short y, float magnitude)
{
    useActiveView();
	Vector2F cursor(x, y);
	unsigned num_active = _activeIndices.length();
	for(unsigned i =0; i < num_active; i++) {
		MMatrix space = worldizeSpace(_spaces[_activeIndices[i]]);
		MPoint pos(space(3, 0), space(3, 1), space(3, 2));
		short viewx, viewy;
		_viewport.worldToView (pos, viewx, viewy);
		Vector2F pscrn(viewx, viewy);
		float weight = pscrn.distantTo(cursor);
		weight = 1.f - weight/128.f;
			
		if(weight>0) {
			float disp = 1.f + magnitude * weight;
			space(0, 0) *= disp;
			space(0, 1) *= disp;
			space(0, 2) *= disp;
			space(1, 0) *= disp;
			space(1, 1) *= disp;
			space(1, 2) *= disp;
			space(2, 0) *= disp;
			space(2, 1) *= disp;
			space(2, 2) *= disp;
			_spaces[_activeIndices[i]] = localizeSpace(space);
		}	
	}
}

void ProxyViz::adjustPosition(short start_x, short start_y, short last_x, short last_y, float clipNear, float clipFar, Matrix44F & mat)
{
    useActiveView();
	MPoint toNear, toFar;
	_viewport.viewToWorld ( last_x, last_y, toNear, toFar );
	
	MPoint fromNear, fromFar;
	_viewport.viewToWorld ( start_x, start_y, fromNear, fromFar );
	
	MVector dispNear = toNear - fromNear;
	MVector dispFar = toFar - fromFar;
	
	Vector3F a(toNear.x, toNear.y, toNear.z);
	Vector3F b(toFar.x, toFar.y, toFar.z);
	Ray r(a, b);
	
	Vector3F v0(dispNear.x, dispNear.y, dispNear.z);
	Vector3F v1(dispFar.x, dispFar.y, dispFar.z);
	
	movePlant(r, v0, v1, clipNear, clipFar);
}

void ProxyViz::adjustRotation(short x, short y, float magnitude, short axis, float noise)
{	
    useActiveView();	
	Vector2F cursor(x, y);
	unsigned num_active = _activeIndices.length();
	for(unsigned i =0; i < num_active; i++) {
		MMatrix space = worldizeSpace(_spaces[_activeIndices[i]]);
		MPoint pos(space(3, 0), space(3, 1), space(3, 2));
		short viewx, viewy;
		_viewport.worldToView (pos, viewx, viewy);
		Vector2F pscrn(viewx, viewy);
		float weight = pscrn.distantTo(cursor);
		weight = 1.f - weight/128.f;
			
		if(weight>0)
		{
			Vector3F first(space(0,0), space(0,1), space(0,2));
			Vector3F second(space(1,0), space(1,1), space(1,2));
			Vector3F third(space(2,0), space(2,1), space(2,2));
			const float scale = first.length();
			first.normalize();
			second.normalize();
			third.normalize();
			if(axis == 0) {
				second.rotateAroundAxis(first, magnitude * (1.f + noise * (float(random()%291) / 291.f - .5f)) * weight);
				third = first.cross(second);				
			}
			else if(axis == 1) {
				first.rotateAroundAxis(second, magnitude * (1.f + noise * (float(random()%347) / 347.f - .5f)) * weight);
				third = first.cross(second);
			}
			else {
				first.rotateAroundAxis(third, magnitude * (1.f + noise * (float(random()%117) / 117.f - .5f)) * weight);
				second = third.cross(first);				
			}
			
			first.normalize();
			second.normalize();
			third.normalize();
			first *= scale;
			second *= scale;
			third *= scale;
			
			space(0, 0) = first.x;
			space(0, 1) = first.y;
			space(0, 2) = first.z;
			space(1, 0) = second.x;
			space(1, 1) = second.y;
			space(1, 2) = second.z;
			space(2, 0) = third.x;
			space(2, 1) = third.y;
			space(2, 2) = third.z;
			_spaces[_activeIndices[i]] = localizeSpace(space);
		}	
	}
}

void ProxyViz::adjustLocation(short start_x, short start_y, short last_x, short last_y, float clipNear, float clipFar, Matrix44F & mat, short axis, float noise)
{	
    useActiveView();	
	MPoint toNear, toFar;
	_viewport.viewToWorld ( last_x, last_y, toNear, toFar );
	
	MPoint fromNear, fromFar;
	_viewport.viewToWorld ( start_x, start_y, fromNear, fromFar );
	
	MVector dispNear = toNear - fromNear;
	MVector dispFar = toFar - fromFar;
	
	Vector3F pp;
	Vector2F cursor(start_x, start_y);
	MVector disp;
	
	float direction = last_x - start_x - last_y + start_y;
	if(direction < 0.f)
		direction = 1.f;
	else
		direction = -1.f;
	
	unsigned num_active = _activeIndices.length();
	for(unsigned i =0; i < num_active; i++) {
		MMatrix space = worldizeSpace(_spaces[_activeIndices[i]]);
		MPoint pos(space(3, 0), space(3, 1), space(3, 2));
		short viewx, viewy;
		_viewport.worldToView (pos, viewx, viewy);
		Vector2F pscrn(viewx, viewy);
		float weight = pscrn.distantTo(cursor);
		weight = 1.f - weight/128.f;
			
		if(weight>0)
		{
			pp = Vector3F(space(3,0), space(3,1), space(3,2));
			pp = mat.transform(pp);
			
			if(pp.z > clipNear) {
				
				disp = dispNear + (dispFar - dispNear)*(pp.z-clipNear)/(clipFar-clipNear);
			
				Vector3F first(space(0,0), space(0,1), space(0,2));
				Vector3F second(space(1,0), space(1,1), space(1,2));
				Vector3F third(space(2,0), space(2,1), space(2,2));
				Vector3F pos(space(3,0), space(3,1), space(3,2));
				
				first.normalize();
				second.normalize();
				third.normalize();
				if(axis == 0) {
					pos += first * direction * disp.length() * (1.f + noise * (float(random()%291) / 291.f - .5f)) * weight;				
				}
				else if(axis == 1) {
					pos += second * direction * disp.length() * (1.f + noise * (float(random()%347) / 347.f - .5f)) * weight;
				}
				else {
					pos += third * direction * disp.length() * (1.f + noise * (float(random()%117) / 117.f - .5f)) * weight;		
				}
				
				space(3, 0) = pos.x;
				space(3, 1) = pos.y;
				space(3, 2) = pos.z;
				_spaces[_activeIndices[i]] = localizeSpace(space);
			}
		}	
	}
}

MMatrix ProxyViz::getActiveBox(unsigned idx) const
{ return worldizeSpace(_spaces[_activeIndices[idx]]); }

int ProxyViz::getActiveIndex(unsigned idx) const
{ return _activeIndices[idx]; }

void ProxyViz::setActiveBox(unsigned idx, const MMatrix & mat)
{ _spaces[_activeIndices[idx]] = localizeSpace(mat); }

unsigned ProxyViz::getNumActiveBoxes() const
{ return _activeIndices.length(); }

void ProxyViz::pressToSave()
{
	MObject thisNode = thisMObject();
	MPlug plugc( thisNode, acachename );
	const MString filename = plugc.asString();
	if(filename != "")
	    saveExternal(replaceEnvVar(filename).c_str());
	else 
		AHelper::Info<int>("ProxyViz error empty external cache filename", 0);
}

void ProxyViz::pressToLoad()
{
	MObject thisNode = thisMObject();
	MPlug plugc( thisNode, acachename );
	const MString filename = plugc.asString();
	if(filename != "")
		loadExternal(replaceEnvVar(filename).c_str());
	else 
		AHelper::Info<int>("ProxyViz error empty external cache filename", 0);
}

void ProxyViz::calculateLOD(const MMatrix & cameraInv, const float & h_fov, const float & aspectRatio, const float & detail, const int & enableViewFrustumCulling)
{	
    float longest = defBox().getLongestDistance();
	int portW, portH;

	if(fHasView) {
		MString srenderer;
		_viewport.getRendererString (srenderer);
		portW = _viewport.portWidth();
		portH = _viewport.portHeight();
	}
	else {
		MGlobal::displayInfo("proxy viz has no renderer");
		return;
	}
				
	unsigned num_box = _details.length();
	
	if(enableViewFrustumCulling == 0) {
		for(unsigned i =1; i < num_box; i++) {
			_details[i] = 1.f;
		}
		return;
	}
	for(unsigned i =1; i < num_box; i++) {
			
		const MMatrix space = worldizeSpace(_spaces[i]);
		const MVector pos(space(3,0), space(3,1), space(3,2)); 
			
		const MVector vx(space(0,0), space(0,1), space(0,2));
		const double sz = vx.length();
		
		const MPoint pcam = MPoint(pos) * cameraInv;
		
		const double depth = -pcam.z;
		double h_max = depth * h_fov;
		double h_min = -h_max;
		double v_max = h_max * aspectRatio;
		double v_min = -v_max;
		
		float realdetail = -1.f;
		
		if(depth < 0.0) {
			if(pcam.distanceTo(MPoint(0,0,0)) < (longest * sz * 2))
				realdetail = 10;
		}
		else {
			if(pcam.x + longest * sz < h_min || pcam.x - longest * sz > h_max || pcam.y + longest * sz < v_min || pcam.y - longest * sz > v_max) {
				if(pcam.distanceTo(MPoint(0,0,0)) < (longest * sz * 2))
					realdetail = 10;
			}
			else {
				realdetail = detail * sz / (h_max + 10e-6);
				if(fHasView) {
					if(pcam.x > h_min && pcam.x < h_max && pcam.y > v_min && pcam.y < v_max) {
						if(!isBoxInView(pos, longest * sz, 0, 0, portW, portH)) 
							realdetail = -1.f;
					}
				}
			}
		}

		if(realdetail > 10e7)
			realdetail = 10e7;
		
		if(realdetail > _details[i])
			_details[i] = realdetail;
	}
}

void ProxyViz::setCullMesh(MDagPath mesh)
{
	MGlobal::displayInfo(MString("proxy viz uses blocker: ") + mesh.fullPathName());
	if(fCuller)
		fCuller->setMesh(mesh);
}

void ProxyViz::updateWorldSpace()
{
	MObject thisNode = thisMObject();
	MDagPath thisPath;
	MDagPath::getAPathTo(thisNode, thisPath);
	_worldSpace = thisPath.inclusiveMatrix();
	_worldInverseSpace = thisPath.inclusiveMatrixInverse();
}

MMatrix ProxyViz::localizeSpace(const MMatrix & s) const
{
	MMatrix m = s;
	m *= _worldInverseSpace;
	return m;
}

MMatrix ProxyViz::worldizeSpace(const MMatrix & s) const
{
	MMatrix m = s;
	m *= _worldSpace;
	return m;
}

void ProxyViz::useActiveView()
{ _viewport = M3dView::active3dView(); }

char ProxyViz::hasDisplayMesh() const
{
	MPlug pm(thisMObject(), ainmesh);
	if(!pm.isConnected())
		return 0;
		
	if(fDisplayMesh == MObject::kNullObj)
		return 0;

	return 1;
}

const MMatrix & ProxyViz::worldSpace() const
{ return _worldSpace; }

std::string ProxyViz::replaceEnvVar(const MString & filename) const
{
    EnvVar var;
	std::string sfilename(filename.asChar());
	if(var.replace(sfilename))
	    MGlobal::displayInfo(MString("substitute file path "+filename+" to ")+sfilename.c_str());
	return sfilename;
}

MStatus ProxyViz::connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == acameraspace) m_hasCamera = true;
	//AHelper::Info<MString>("connect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

MStatus ProxyViz::connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc )
{
	if(plug == acameraspace) m_hasCamera = false;
	//AHelper::Info<MString>("disconnect", plug.name());
	return MPxLocatorNode::connectionMade (plug, otherPlug, asSrc );
}

void ProxyViz::vizViewFrustum(MObject & thisNode)
{
	MPlug hfaplg(thisNode, ahapeture);
	float hfa = hfaplg.asFloat();
	MPlug vfaplg(thisNode, ahapeture);
	float vfa = vfaplg.asFloat();
	MPlug flplg(thisNode, afocallength);
	float fl = flplg.asFloat();
	float hfov = hfa * 0.5 / ( fl * 0.03937 );
	float aspectRatio = vfa / hfa;
	
	MPlug matplg(thisNode, acameraspace);
	MObject matobj;
	matplg.getValue(matobj);
	MFnMatrixData matdata(matobj);
    MMatrix cameramat = matdata.matrix(); 
	
	Matrix44F cameraSpace;
	AHelper::ConvertToMatrix44F(cameraSpace, cameramat);
	Matrix44F worldInvSpace;
	AHelper::ConvertToMatrix44F(worldInvSpace, _worldInverseSpace);
	drawViewFrustum(cameraSpace, worldInvSpace, hfov, aspectRatio);
}
//:~