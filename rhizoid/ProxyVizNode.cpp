#include <Vector2F.h>
#include "proxyVizNode.h"
#include <maya/MVector.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MDataHandle.h>
#include <maya/MColor.h>
#include <maya/MDistance.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MVectorArray.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MEulerRotation.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MIntArray.h>
#include <maya/MPointArray.h>
#include <maya/MFnMeshData.h>
#include <maya/MItMeshPolygon.h>
#include <PseudoNoise.h>
#include <EnvVar.h>
#include <AHelper.h>
#include <fstream> 

static int indexByValue(const MIntArray &arr, int value) 
{
	unsigned numElement = arr.length();
	for(unsigned i = 0; i < numElement; i++) {
		if(arr[i] == value)
			return i;
	}
	return -1;
}

static void matrix_as_array(const MMatrix &space, double *mm)
{
	mm[0] = space(0,0);
	mm[1] = space(0,1);
	mm[2] = space(0,2);
	mm[3] = space(0,3);
	mm[4] = space(1,0);
	mm[5] = space(1,1);
	mm[6] = space(1,2);
	mm[7] = space(1,3);
	mm[8] = space(2,0);
	mm[9] = space(2,1);
	mm[10] = space(2,2);
	mm[11] = space(2,3);
	mm[12] = space(3,0);
	mm[13] = space(3,1);
	mm[14] = space(3,2);
	mm[15] = space(3,3);

}

static void scale_matrix(const Vector3F &scale, float *mm)
{
	mm[0] = scale.x;
	mm[5] = scale.y;
	mm[10] = scale.z;
	mm[15] = 1.f;
	mm[1] = mm[2] = mm[3] = mm[4] = mm[6] = mm[7] = mm[8] = mm[9] = mm[11] = mm[12] = mm[13] = mm[14] = 0.f;
	
}

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

ProxyViz::ProxyViz() : _firstLoad(1), fHasView(0), fVisibleTag(0), fCuller(0),
m_toSetGrid(true)
{
	MMatrix m;
	addABox(m);
}

ProxyViz::~ProxyViz() 
{
    if(fVisibleTag)
        delete[] fVisibleTag;
}

MStatus ProxyViz::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug == outValue ) {
		
		updateWorldSpace();
		
		MStatus status;

		MMatrix cameraInv;
		MDataHandle cameradata = block.inputValue(acameraspace, &status);
        if(status) {
			cameraInv = cameradata.asMatrix();
		}
		
		fDisplayMesh = block.inputValue( ainmesh ).asMesh();

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
		MString bakename =  block.inputValue( adumpname ).asString();
		MString filename =  block.inputValue( acachename ).asString();	
		const double percentage = block.inputValue(aconvertPercentage).asDouble();
		m_materializePercentage = percentage;

		if(_firstLoad) {
			if(filename != "") {
			    loadCache(replaceEnvVar(filename).c_str());
			}	
			_firstLoad = 0;
		}
        
		MArrayDataHandle groundArray = block.inputArrayValue(agroundMesh );
        updateGround(groundArray );
		
		unsigned num_box = _spaces.length();
		_details.setLength(num_box);
		_randNums.setLength(num_box);
		
		if(fVisibleTag)
		    delete[] fVisibleTag;
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
	
	Vector3F multiplier;
	MPlug mutxplug( thisNode, axmultiplier);
	multiplier.x = mutxplug.asFloat();
	
	MPlug mutyplug( thisNode, aymultiplier);
	multiplier.y = mutyplug.asFloat();
	
	MPlug mutzplug( thisNode, azmultiplier);
	multiplier.z = mutzplug.asFloat();	
	
	float mScale[16];
	scale_matrix(multiplier, mScale);
	
	MPlug matplg(thisNode, acameraspace);
	MObject matobj;
	matplg.getValue(matobj);
	MFnMatrixData matdata(matobj);
    MMatrix cameramat = matdata.matrix(); 
	cameramat = cameramat.inverse();
	
	MPlug hfaplg(thisNode, ahapeture);
	float hfa = hfaplg.asFloat();
	MPlug vfaplg(thisNode, ahapeture);
	float vfa = vfaplg.asFloat();
	MPlug flplg(thisNode, afocallength);
	float fl = flplg.asFloat();
	float hfov = hfa * 0.5 / ( fl * 0.03937 );
	float aspectRatio = vfa / hfa;
	
	_viewport = view;
	fHasView = 1;

	view.beginGL();
	
	drawViewFrustum(cameramat, hfov, aspectRatio);

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
	drawGrid();

	if ( ( style == M3dView::kFlatShaded ) || 
		    ( style == M3dView::kGouraudShaded ) ) {		
		drawPlants();
	}
	else {
		drawWiredPlants();
	}

	drawSelected(mScale);
	drawGround();
	view.endGL();
}

bool ProxyViz::isBounded() const
{ return true; }

MBoundingBox ProxyViz::boundingBox() const
{   
	BoundingBox bbox = defBox();
	if(numPlants() > 0) bbox.expandBy(gridBoundingBox() );
	
	Vector3F vmin = bbox.getMin();
	Vector3F vmax = bbox.getMax();
	MPoint corner1(vmin.x, vmin.y, vmin.z);
	MPoint corner2(vmax.x, vmax.y, vmax.z);

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
    
	attributeAffects(acameraspace, outValue);
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

void ProxyViz::drawSelected(float mScale[16])
{
	double mm[16];
	glDisable(GL_DEPTH_TEST);
	unsigned num_active = getNumActiveBoxes();
	for(unsigned i =0; i < num_active; i++) {
		glPushMatrix();

		const MMatrix space = _spaces[_activeIndices[i]];
		matrix_as_array(space, mm);
		
		glMultMatrixd(mm);
		glMultMatrixf(mScale);
		draw_coordsys();
		glPopMatrix();
	}		
	glEnable(GL_DEPTH_TEST);
}

void ProxyViz::drawViewFrustum(const MMatrix & cameraSpace, const float & h_fov, const float & aspectRatio)
{
	float fnear = -1.f;
	float ffar = -250000.f;
	float nearRight = fnear * h_fov;
	float nearLeft = -nearRight;
	float nearUp = nearRight * aspectRatio;
	float nearBottom = -nearUp;
	float farRight = ffar * h_fov;
	float farLeft = -farRight;
	float farUp = farRight * aspectRatio;
	float farBottom = -farUp;
	MPoint clipNear[4];
	MPoint clipFar[4];
	
	clipNear[0] = MPoint(nearLeft, nearBottom, fnear);
	clipNear[1] = MPoint(nearRight, nearBottom, fnear);
	clipNear[2] = MPoint(nearRight, nearUp, fnear);
	clipNear[3] = MPoint(nearLeft, nearUp, fnear);
	
	clipFar[0] = MPoint(farLeft, farBottom, ffar);
	clipFar[1] = MPoint(farRight, farBottom, ffar);
	clipFar[2] = MPoint(farRight, farUp, ffar);
	clipFar[3] = MPoint(farLeft, farUp, ffar);
	
	for(int i=0; i < 4; i++) {
		clipNear[i] *= cameraSpace;
		clipNear[i] *= _worldInverseSpace;
		clipFar[i] *= cameraSpace;
		clipFar[i] *= _worldInverseSpace;
	}
	
	MPoint p;
	glBegin(GL_LINES);
	for(int i=0; i < 4; i++) {
		p = clipNear[i];
		glVertex3f(p.x, p.y, p.z);
		p = clipFar[i];
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
	
}

void ProxyViz::addABox(const MMatrix & m)
{
	const MMatrix localm = localizeSpace(m);
	_spaces.append(localm);
	_activeIndices.append(_spaces.length() - 1);
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

float longestDimension(const Vector3F & bbmax, const Vector3F & bbmin)
{
	Vector3F bb = bbmax - bbmin;
	float longest = bb.x;
	if(bb.y > longest) longest = bb.y;
	if(bb.z > longest) longest = bb.z;
	return longest;
}

void ProxyViz::selectBoxesInView(short xmin, short ymin, short xmax, short ymax, MGlobal::ListAdjustment selectionMode)
{	
    useActiveView();
	float longest = defBox().getLongestDistance();

	if(selectionMode == MGlobal::kReplaceList)
		_activeIndices.clear();
		
	unsigned num_box = _spaces.length();
	for(unsigned i =1; i < num_box; i++) {
		const MMatrix space = worldizeSpace(_spaces[i]);

		const MPoint pos(space(3,0), space(3,1), space(3,2));
		const MVector side(space(0,0), space(0,1), space(0,2)); 
					
		if(isBoxInView(pos, longest * side.length(), xmin, ymin, xmax, ymax)) {
			if(selectionMode == MGlobal::kReplaceList) {
				_activeIndices.append(i);
			}
			else {
				int found = indexByValue(_activeIndices, i);
                                if(selectionMode == MGlobal::kXORWithList) {
                                    if(found < 0)
                                        _activeIndices.append(i);
                                    else
                                        _activeIndices.remove(found);
                                }
                                else if(selectionMode == MGlobal::kRemoveFromList) {
                                    if(found > -1)
                                        _activeIndices.remove(found);
                                }
                                else if(selectionMode == MGlobal::kAddToList) {
                                    if(found < 0)
                                        _activeIndices.append(i);
                                }
                        }

		}
	}
}

void ProxyViz::removeBoxesInView(short xmin, short ymin, short xmax, short ymax, const float & threshold)
{
    useActiveView();
	float longest = defBox().getLongestDistance();

	_activeIndices.clear();
	unsigned num_box = _spaces.length();
	for(unsigned i =1; i < num_box; i++) {
		const MMatrix space = worldizeSpace(_spaces[i]);

		const MPoint pos(space(3,0), space(3,1), space(3,2));
		const MVector side(space(0,0), space(0,1), space(0,2)); 
		
				
		if(isBoxInView(pos, longest * side.length(), xmin, ymin, xmax, ymax)) {
			float noi = float(random()%191)/191.f;
			if(noi < threshold) {
				_spaces.remove(i);
				i--;
				num_box--;
			}
		}
	}
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
			
		if(weight>0)
		{
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

void ProxyViz::adjustPosition(short start_x, short start_y, short last_x, short last_y, float clipNear, float clipFar, Matrix44F & mat, MFnMesh & mesh)
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
				
				MPoint movedp = MPoint(space(3,0), space(3,1), space(3,2)) + MVector(disp.x, disp.y, disp.z) * weight;
				
				MPoint closestP;
				MVector closestN;
				mesh.getClosestPointAndNormal (movedp, closestP, closestN, MSpace::kWorld);
	
				space(3, 0) =  closestP.x;
				space(3, 1) =  closestP.y;
				space(3, 2) =  closestP.z;

				_spaces[_activeIndices[i]] = localizeSpace(space);
			}
		}	
	}
}

void ProxyViz::smoothPosition(short start_x, short start_y, short last_x, short last_y, float clipNear, float clipFar, Matrix44F & mat, MFnMesh & mesh)
{
	unsigned num_active = _activeIndices.length();
	if(num_active < 2) return;
	MVectorArray wp;
	wp.setLength(num_active);
	for(unsigned i =0; i < num_active; i++) {
		MMatrix space = worldizeSpace(_spaces[_activeIndices[i]]);
		wp[i] = MVector(space(3,0), space(3,1), space(3,2));
	}
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

void ProxyViz::snapByIntersection(MFnMesh &mesh)
{
    MStatus stat;
    unsigned num_active = _activeIndices.length();
	for(unsigned i =0; i < num_active; i++) {
	    MMatrix space = worldizeSpace(_spaces[_activeIndices[i]]);
		MPoint pos(space(3, 0), space(3, 1), space(3, 2));
		const MVector dir(space(1, 0), space(1, 1), space(1, 2));
		const MVector invdir = dir * -1.0;
	    
		MPointArray Aphit;
        MIntArray Aihit;
        MPoint hit;
        MVector nor;
        char validHit = 0;
        if(mesh.intersect (pos, dir, Aphit, 0, MSpace::kWorld, &Aihit, &stat)) {
            hit = Aphit[0];
            mesh.getPolygonNormal (Aihit[0], nor,  MSpace::kWorld );
            if(nor * dir > 0.0)
                validHit = 1;   
        }
        
        if(!validHit) {
            if(mesh.intersect (pos, invdir, Aphit, 0, MSpace::kWorld, &Aihit, &stat)) {
                hit = Aphit[0];
                mesh.getPolygonNormal (Aihit[0], nor,  MSpace::kWorld );
                if(nor * dir > 0.0)
                    validHit = 1;  
            }
        }
        
        if(validHit) {
            space(3, 0) = hit.x;
			space(3, 1) = hit.y;
			space(3, 2) = hit.z;
			_spaces[_activeIndices[i]] = localizeSpace(space);
        }
	}
}

MMatrix ProxyViz::getActiveBox(unsigned idx) const
{
	return worldizeSpace(_spaces[_activeIndices[idx]]);
}

int ProxyViz::getActiveIndex(unsigned idx) const
{
	return _activeIndices[idx];
}

void ProxyViz::setActiveBox(unsigned idx, const MMatrix & mat)
{
	_spaces[_activeIndices[idx]] = localizeSpace(mat);
}

unsigned ProxyViz::getNumActiveBoxes() const
{
	return _activeIndices.length();
}

void ProxyViz::loadCache(const char* filename)
{
    MGlobal::displayInfo("proxy viz loading...");
	std::ifstream chFile;
	chFile.open(filename, std::ios_base::in | std::ios_base::binary);
	if(!chFile.is_open()) {
		MGlobal::displayWarning(MString("proxy viz cannot open file: ") + filename);
		return;
	}
	
	chFile.seekg (0, ios::end);

	if(chFile.tellg() < 4 + 4 * 16) {
		MGlobal::displayWarning(MString("proxy viz skipped empty file: ") + filename);
		chFile.close();
		return;
	}
	
	chFile.seekg (0, ios::beg);
	int numRec;
	chFile.read((char*)&numRec, sizeof(int));
	MGlobal::displayInfo(MString("proxy viz read recond count ") + numRec);
	float *data = new float[numRec * 16];
	chFile.read((char*)data, sizeof(float) * numRec * 16);
	chFile.close();
	
	_spaces.setLength(numRec);
	for(int i=0; i < numRec; i++) {
		MMatrix space;
		const int ii = i * 16;
		space(0, 0) = data[ii];
		space(0, 1) = data[ii+1];
		space(0, 2) = data[ii+2];
		space(1, 0) = data[ii+4];
		space(1, 1) = data[ii+5];
		space(1, 2) = data[ii+6];
		space(2, 0) = data[ii+8];
		space(2, 1) = data[ii+9];
		space(2, 2) = data[ii+10];
		space(3, 0) = data[ii+12];
		space(3, 1) = data[ii+13];
		space(3, 2) = data[ii+14];
		
		_spaces[i] = space;
	}
	MGlobal::displayInfo(MString("proxy viz read cache from ") + filename);
	delete[] data;
}

void ProxyViz::saveCache(const char* filename)
{
	std::ofstream chFile;
	chFile.open(filename, std::ios_base::out | std::ios_base::binary);
	if(!chFile.is_open()) {
		MGlobal::displayWarning(MString("proxy viz cannot open file: ") + filename);
		return;
	}
	int numRec = _spaces.length();
	MGlobal::displayInfo(MString("proxy viz write recond count ") + numRec);
	chFile.write((char*)&numRec, sizeof(int));
	float *data = new float[numRec * 16];
	for(int i=0; i < numRec; i++) {
		const MMatrix space = _spaces[i];
		const int ii = i * 16;
		data[ii] = space(0, 0);
		data[ii+1] = space(0, 1);
		data[ii+2] = space(0, 2);
		data[ii+4] = space(1, 0);
		data[ii+5] = space(1, 1);
		data[ii+6] = space(1, 2);
		data[ii+8] = space(2, 0);
		data[ii+9] = space(2, 1);
		data[ii+10] = space(2, 2);
		data[ii+12] = space(3, 0);
		data[ii+13] = space(3, 1);
		data[ii+14] = space(3, 2);
	}
	chFile.write((char*)data, sizeof(float) * numRec * 16);
	chFile.close();
	MGlobal::displayInfo(MString("Well done! Proxy saved to ") + filename);
	delete[] data;
}

void ProxyViz::pressToSave()
{
	MObject thisNode = thisMObject();
	MPlug plugc( thisNode, acachename );
	const MString filename = plugc.asString();
	if(filename != "") {
	    saveCache(replaceEnvVar(filename).c_str());
	}
}

void ProxyViz::pressToLoad()
{
	MObject thisNode = thisMObject();
	MPlug plugc( thisNode, acachename );
	const MString filename = plugc.asString();
	if(filename != "")
		loadCache(replaceEnvVar(filename).c_str());
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

void ProxyViz::removeAllBoxes()
{
	_activeIndices.clear();
	_spaces.clear();
	MMatrix m;
	addABox(m);
}

void ProxyViz::setCullMesh(MDagPath mesh)
{
	MGlobal::displayInfo(MString("proxy viz uses blocker: ") + mesh.fullPathName());
	if(fCuller)
		fCuller->setMesh(mesh);
}

void ProxyViz::bakePass(const char* filename, const MVectorArray & position, const MVectorArray & scale, const MVectorArray & rotation)
{
	std::ofstream chFile;
	chFile.open(filename, std::ios_base::out | std::ios_base::binary);
	if(!chFile.is_open()) {
		MGlobal::displayWarning(MString("proxy viz cannot open file: ") + filename);
		return;
	}
	int numRec = position.length();
	MGlobal::displayInfo(MString("proxy viz write bake recond count ") + numRec);
	chFile.write((char*)&numRec, sizeof(int));
	float *data = new float[numRec * 9];
	for(int i=0; i < numRec; i++) {
		const int ii = i * 9;
		const MVector pos = position[i];
		const MVector scl = scale[i];
		const MVector rot = rotation[i];
		data[ii] = pos.x;
		data[ii+1] = pos.y;
		data[ii+2] = pos.z;
		data[ii+3] = scl.x;
		data[ii+4] = scl.y;
		data[ii+5] = scl.z;
		data[ii+6] = rot.x;
		data[ii+7] = rot.y;
		data[ii+8] = rot.z;
	}
	chFile.write((char*)data, sizeof(float) * numRec * 9);
	chFile.close();
	MGlobal::displayInfo(MString("Well done! Proxy pass saved to ") + filename);
	delete[] data;
}

void ProxyViz::updateWorldSpace()
{
	MObject thisNode = thisMObject();
	MDagPath thisPath;
	MDagPath::getAPathTo(thisNode, thisPath);
	_worldSpace = thisPath.inclusiveMatrix();
	_worldInverseSpace = thisPath.inclusiveMatrixInverse();
	//MGlobal::displayInfo(MString("ws p ")+_worldSpace(3,0)+" "+_worldSpace(3,1)+" "+_worldSpace(3,2));	
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
{
     _viewport = M3dView::active3dView();
}

char ProxyViz::hasDisplayMesh() const
{
	MPlug pm(thisMObject(), ainmesh);
	if(!pm.isConnected())
		return 0;
		
	if(fDisplayMesh == MObject::kNullObj)
		return 0;

	return 1;
}

void ProxyViz::drawSolidMesh(MItMeshPolygon & iter)
{
	iter.reset();
	for(; !iter.isDone(); iter.next()) {
		int vertexCount = iter.polygonVertexCount();
		glBegin(GL_POLYGON);
		for(int i=0; i < vertexCount; i++) {
			MPoint p = iter.point (i);
			MVector n;
			iter.getNormal(i, n);
			glNormal3f(n.x, n.y, n.z);
			glVertex3f(p.x, p.y, p.z);
		}
		glEnd();
	}
}

void ProxyViz::drawWireMesh(MItMeshPolygon & iter)
{
	iter.reset();
	glBegin(GL_LINES);
	for(; !iter.isDone(); iter.next()) {
		int vertexCount = iter.polygonVertexCount();
		
		for(int i=0; i < vertexCount-1; i++) {
			MPoint p = iter.point (i);
			glVertex3f(p.x, p.y, p.z);
			p = iter.point (i+1);
			glVertex3f(p.x, p.y, p.z);
		}		
	}
	glEnd();
}

const MMatrixArray & ProxyViz::spaces() const
{
    return _spaces; 
}

const MMatrixArray ProxyViz::spaces(int groupCount, int groupId, MIntArray & ppNums) const
{
    MMatrixArray res;
    const unsigned num_box = _spaces.length();
    for(unsigned i =1; i < num_box; i++) {	
        if(groupCount > 1) {
            int grp = _randNums[i] % groupCount;
            if(grp != groupId)
                continue;
        }
			
        if(m_materializePercentage < 1.0) {
            double dart = ((double)(rand()%497))/497.0;
            if(dart > m_materializePercentage) continue;
        }
			   
        const MMatrix space = _spaces[i];
        res.append(space);
        ppNums.append(_randNums[i]);
    }
    return res;
}

const MMatrix & ProxyViz::worldSpace() const
{
    return _worldSpace;
}

std::string ProxyViz::replaceEnvVar(const MString & filename) const
{
    EnvVar var;
	std::string sfilename(filename.asChar());
	if(var.replace(sfilename))
	    MGlobal::displayInfo(MString("substitute file path "+filename+" to ")+sfilename.c_str());
	return sfilename;
}
//:~
