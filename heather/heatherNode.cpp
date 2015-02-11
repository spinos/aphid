///:
// heatherNode.cpp
// Zhang Jian
// 07/12/05

#include "heatherNode.h"
#include <maya/MItDag.h>
#include <maya/MItDependencyNodes.h>
#include <maya/MPlugArray.h>
#include <maya/MFnMesh.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MPointArray.h>
#include <SHelper.h>
MTypeId heatherNode::id( 0x7065d6 );

MObject heatherNode::amatrix;
MObject heatherNode::anear;
MObject heatherNode::afar;
MObject heatherNode::ahapeture;
MObject heatherNode::avapeture;
MObject heatherNode::afocallength;
MObject heatherNode::aorthographic;
MObject heatherNode::aorthographicwidth;
MObject heatherNode::adepthImageName;
MObject heatherNode::aenableMultiFrames;
MObject heatherNode::aframeNumber;
MObject heatherNode::aframePadding;
MObject heatherNode::ablockSetName;
MObject heatherNode::outValue;

heatherNode::heatherNode() 
{
    m_needLoadImage = 0;
    m_exr = 0;
	m_framebuffer = 0;
	// m_blockVs = 0;
	// m_blockTriIndices = 0;
}
heatherNode::~heatherNode() 
{
    if(m_exr) delete m_exr;
	if(m_framebuffer) delete m_framebuffer;
	// if(m_blockVs) delete[] m_blockVs;
	// if(m_blockTriIndices) delete[] m_blockTriIndices;
}

MStatus heatherNode::compute( const MPlug& plug, MDataBlock& block )
{ 
    if( plug == outValue ) {
        MString filename = block.inputValue(adepthImageName).asString();
        int frame = block.inputValue(aframeNumber).asInt();
		int padding = block.inputValue(aframePadding).asInt();
		bool enableSequence = block.inputValue(aenableMultiFrames).asBool();
        preLoadImage(filename.asChar(), frame, padding, enableSequence);
		MString setname = block.inputValue(ablockSetName).asString();
		cacheBlocks(setname);
        MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set(0.0);
    }
	return MS::kSuccess;
}

void heatherNode::preLoadImage(const char * name, int frame, int padding, bool useImageSequence)
{
    std::string fileName(name);
    if(fileName.size() < 3) return;
	
	if(useImageSequence)
		SHelper::changeFrameNumber(fileName, frame, padding);
		
	if(!m_exr) m_exr = new ZEXRImage(fileName.c_str(), false);
	else m_exr->open(fileName.c_str());
    if(!m_exr->isOpened()) {
		MGlobal::displayInfo(MString("cannot open image ") + fileName.c_str());
		return;
	}

	// std::vector<std::string> names;
    // ZEXRImage::listExrChannelNames(fileName, names);
    
    // std::vector<std::string>::const_iterator it = names.begin();
    // for(;it != names.end();++it) MGlobal::displayWarning((*it).c_str());

    if(!m_exr->isRGBAZ()) {
        MGlobal::displayWarning(MString("image is not RGBAZ format.") + fileName.c_str());
        return;
    }
    
    m_needLoadImage = 1;
}

void heatherNode::cacheBlocks(const MString & setname)
{
	if(setname.length() < 2) return;
	MItDependencyNodes itdag(MFn::kSet);
	MFnDependencyNode fn;
	bool found = 0;
	for(;!itdag.isDone();itdag.next()) {
		MObject	n = itdag.thisNode();
		fn.setObject(n);
		if(fn.name() == setname) {
			// MGlobal::displayInfo(setname+" is "+fn.typeName());
			found = 1;
			break;
		}
	}
	
	if(!found) {
		MGlobal::displayWarning(setname + " cannot be found");
		return;
	}
	
	m_meshes.clear();
	
	// MGlobal::displayInfo(fn.name());
	MStringArray memberNames;
	MGlobal::executeCommand(MString("listConnections ")+fn.name(), memberNames);
	// MGlobal::displayInfo(MString("n connections ")+memberNames.length());
	for(unsigned i=0; i<memberNames.length(); i++) cacheMeshFromNode(memberNames[i]);
	
	/*MStatus status;
	MPlug plgmembers = fn.findPlug("dagSetMembers", &status);
	if(!status) MGlobal::displayInfo("not found plg ");
	
	if( plgmembers.isArray()) MGlobal::displayInfo(plgmembers.name() + " is array");
	MGlobal::displayInfo(plgmembers.name() + " n elem " + plgmembers.numElements());
	MPlugArray plgs;
	
	unsigned i, j;
	for (j = 0; j < plgmembers.numElements (); j++) {
		MPlug elementPlug = plgs[j];
		elementPlug.connectedTo(plgs, false, true);
		MGlobal::displayInfo(MString("n connections ")+plgs.length());
		for(i=0; i<plgs.length(); i++)
			cacheMeshFromNode(plgs[i].node());
	}*/
}

void heatherNode::cacheMeshes()
{
	if(m_meshes.length() < 1) return;
	unsigned nv = 0;
	unsigned ntriv = 0;
	unsigned i;
	for(i=0; i<m_meshes.length(); i++) {
		MFnMesh fmesh(m_meshes[i]);
		nv += fmesh.numVertices();
		MIntArray triangleCounts, triangleVertices;
		fmesh.getTriangles(triangleCounts, triangleVertices);
		ntriv += triangleVertices.length();
	}
	MGlobal::displayInfo(MString("n v ")+nv);
	MGlobal::displayInfo(MString("n tri v ")+ntriv);
/*
	if(m_blockVs) delete[] m_blockVs;
	if(m_blockTriIndices) delete[] m_blockTriIndices;
	
	m_blockVs = new Vector3F[nv];
	m_blockTriIndices = new unsigned[ntriv];
	
	Vector3F *currentV = &m_blockVs[0];
	unsigned *currentI = &m_blockTriIndices[0];
	unsigned iOffset = 0;
	for(i=0; i<m_meshes.length(); i++) {
		MFnMesh fmesh(m_meshes[i]);
		MPointArray vertexArray;
		fmesh.getPoints(vertexArray, MSpace::kWorld);
		for(j=0; j < vertexArray.length(); j++) {
			currentV->set(vertexArray[j].x, vertexArray[j].y, vertexArray[j].z);
			currentV++;
		}
		
		MIntArray triangleCounts, triangleVertices;
		fmesh.getTriangles(triangleCounts, triangleVertices);
		
		for(j=0; j < triangleVertices.length(); j++) {
			*currentI = triangleVertices[j] + iOffset;
			currentI++;
		}
		
		iOffset += fmesh.numVertices();
	}
*/
}

void heatherNode::cacheMeshFromNode(const MString & name)
{
	MItDag itdag(MItDag::kDepthFirst, MFn::kTransform);
	for(;!itdag.isDone();itdag.next()) {
		if(itdag.partialPathName() == name) {
			cacheMeshFromNode(itdag.currentItem());
			return;
		}
	}
}

void heatherNode::cacheMeshFromNode(const MObject & node)
{
	MItDag itdag;
	itdag.reset(node, MItDag::kDepthFirst, MFn::kMesh);
	for(;!itdag.isDone();itdag.next()) {
		// MGlobal::displayInfo(itdag.partialPathName()+"found mesh");
		MDagPath currentPath;
		itdag.getPath(currentPath);
		m_meshes.append(currentPath);
	}
}

void heatherNode::draw( M3dView & view, const MDagPath & /*path*/, 
							 M3dView::DisplayStyle style,
							 M3dView::DisplayStatus status )
{ 

	MObject thisNode = thisMObject();
	MFnDependencyNode nodeFn(thisNode);

	MPlug matPlug = nodeFn.findPlug( "cameraMatrix" );
	
	MObject mat_obj;
	matPlug.getValue( mat_obj );
	MFnMatrixData mat_data( mat_obj );
	MMatrix cmat = mat_data.matrix();
	
	MPoint eye;
	eye *= cmat;
	
	double near_clip, far_clip;
	
	MPlug nearPlug = nodeFn.findPlug( "nearClip" );
	nearPlug.getValue( near_clip );
	near_clip -= 0.001;
	
	MPlug farPlug = nodeFn.findPlug( "farClip" );
	farPlug.getValue( far_clip );
	far_clip += 0.001;
	
	double h_apeture, v_apeture;
	MPlug hfaPlug = nodeFn.findPlug( "horizontalFilmAperture" );
	hfaPlug.getValue( h_apeture );
	
	MPlug vfaPlug = nodeFn.findPlug( "verticalFilmAperture" );
	vfaPlug.getValue( v_apeture );
	
	double fl;
	MPlug flPlug = nodeFn.findPlug( "focalLength" );
	flPlug.getValue( fl );
	
	double orthwidth;
	MPlug owPlug = nodeFn.findPlug( "orthographicWidth" );
	owPlug.getValue( orthwidth );
	
	bool orth;
	MPlug orthPlug = nodeFn.findPlug( "orthographic" );
	orthPlug.getValue( orth );

	double h_fov = h_apeture * 0.5 / ( fl * 0.03937 );
	double v_fov = v_apeture * 0.5 / ( fl * 0.03937 );
	
	float fright = far_clip * h_fov;
	float ftop = far_clip * v_fov;
	
	float nright = near_clip * h_fov;
	float ntop = near_clip * v_fov;
	
	if(orth) fright = ftop = nright = ntop = orthwidth/2.0;
	
	
	MPoint corner_a(fright,ftop,-far_clip);
	corner_a *= cmat;
	
	MPoint corner_b(-fright,ftop,-far_clip);
	corner_b *= cmat;
	
	MPoint corner_c(-fright,-ftop,-far_clip);
	corner_c *= cmat;
	
	MPoint corner_d(fright,-ftop,-far_clip);
	corner_d *= cmat;
	
	MPoint corner_e(nright,ntop,-near_clip);
	corner_e *= cmat;
	
	MPoint corner_f(-nright,ntop,-near_clip);
	corner_f *= cmat;
	
	MPoint corner_g(-nright,-ntop,-near_clip);
	corner_g *= cmat;
	
	MPoint corner_h(nright,-ntop,-near_clip);
	corner_h *= cmat;

	view.beginGL();
	
    if(!m_clamp.isDiagnosed()) {
#ifdef WIN32
        MGlobal::displayInfo("init glext on win32");
		gExtensionInit();
#endif
		std::string log;
		m_clamp.diagnose(log);
		MGlobal::displayInfo(MString("glsl diagnose log: ") + log.c_str());
		glGenTextures(1, &m_bgdCImg);
		glGenTextures(1, &m_depthImg);
		glGenTextures(1, &m_colorImg);
						
		m_depth.diagnose(log);
	}	

    if(!m_exr) return;
    
    const float imageAspectRatio = m_exr->aspectRation();
	
	GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];
    glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
    glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
    glGetIntegerv (GL_VIEWPORT, viewport);
	Matrix44F mmv(mvmatrix);
    Matrix44F mmvinv(mvmatrix); mmvinv.inverse();
    Matrix44F mproj(projmatrix);
    
    const GLint width = viewport[2];
    const GLint portHeight = viewport[3];
    
    int height = width * imageAspectRatio;
    if(height < 2) height = 2;
    if(height > portHeight) height = portHeight;
    
    const float realRatio = (float)height/(float)width;
    
    unsigned char *pixels = new unsigned char[width * height * 4];
    
// 8-bit only?
// only GL_DEPTH_COMPONENT works
    glReadPixels(0, (portHeight - height)/2, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    
    //MGlobal::displayInfo(MString("dep")+pixels[128]);
    
    glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_bgdCImg);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB );
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL );
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	
	delete[] pixels;
	
	if(m_needLoadImage) {
	    glActiveTexture(GL_TEXTURE1);
	    glBindTexture(GL_TEXTURE_2D, m_colorImg);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
#ifdef WIN32
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, m_exr->getWidth(), m_exr->getHeight(), 0, GL_RGBA, GL_HALF_FLOAT, m_exr->_pixels);
#else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, m_exr->getWidth(), m_exr->getHeight(), 0, GL_RGBA, GL_HALF_FLOAT_ARB, m_exr->_pixels);
#endif	
	    glActiveTexture(GL_TEXTURE2);
	    glBindTexture(GL_TEXTURE_2D, m_depthImg);
	    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB );
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL );
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, m_exr->getWidth(), m_exr->getHeight(), 0, GL_RED, GL_FLOAT, m_exr->m_zData);
	    
		// for(int i=0; i < m_exr->getWidth() * m_exr->getHeight(); i+=999) MGlobal::displayInfo(MString("z ")+m_exr->m_zData[i]);
		int fbw = m_exr->getWidth();
		if(fbw > 2048) fbw = 2048;
		int fbh = fbw * m_exr->aspectRation();
		
		if(m_framebuffer) delete m_framebuffer;
		m_framebuffer = new GlFramebuffer(fbw, fbh);
		// if(m_framebuffer->hasFbo()) MGlobal::displayInfo("fbo created");
				
		m_clamp.setTextures(m_framebuffer->colorTexture(), m_bgdCImg,
				m_depthImg, 
				m_colorImg);
				
		m_needLoadImage = 0;
	}
	
	m_framebuffer->begin();
	m_depth.programBegin();
	glColor3f(1,1,1);
	
	drawBackPlane(mproj, mmvinv, realRatio);
	drawBlocks();
	
	m_depth.programEnd();
	m_framebuffer->end();
	
	float tt;
    Vector3F leftP;
    
    Plane pfar(mproj.M(0,3) - mproj.M(0,2), 
               mproj.M(1,3) - mproj.M(1,2),
               mproj.M(2,3) - mproj.M(2,2), 
               mproj.M(3,3) - mproj.M(3,2));
    // MGlobal::displayInfo(MString("proj")+mproj.str().c_str());
    Ray toFar(Vector3F(0,0,0), Vector3F(0,0,-1), 0.f, 1e8);
    
    pfar.rayIntersect(toFar, leftP, tt);
    // MGlobal::displayInfo(MString("far")+leftP.str().c_str());
	m_clamp.setClippings(0.1f, -leftP.z);
	
	glPushAttrib(GL_ALL_ATTRIB_BITS);
    m_clamp.programBegin();

    glDisable(GL_DEPTH_TEST);
    
	drawBackPlane(mproj, mmvinv, realRatio);
    
	m_clamp.programEnd();
	
	glActiveTexture(GL_TEXTURE0);
    glDisable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE1);
    glDisable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE2);
    glDisable(GL_TEXTURE_2D);
	
	glEnable(GL_DEPTH_TEST);
	
	glPopAttrib();
	view.endGL();
	
}

void heatherNode::drawBackPlane(const Matrix44F & mproj, const Matrix44F & mmvinv, const float & aspectRatio)
{
	float tt;
    Vector3F leftP;
    
    Plane pfar(mproj.M(0,3) - mproj.M(0,2), 
               mproj.M(1,3) - mproj.M(1,2),
               mproj.M(2,3) - mproj.M(2,2), 
               mproj.M(3,3) - mproj.M(3,2));
    // MGlobal::displayInfo(MString("proj")+mproj.str().c_str());
    Ray toFar(Vector3F(0,0,0), Vector3F(0,0,-1), 0.f, 1e8);
    
    pfar.rayIntersect(toFar, leftP, tt);
    
    Plane pleft(mproj.M(0,3) + mproj.M(0,0), 
               mproj.M(1,3) + mproj.M(1,0),
               mproj.M(2,3) + mproj.M(2,0), 
               mproj.M(3,3) + mproj.M(3,0));
    
    const float zPlane = leftP.z * .999f;
    Ray toleft(Vector3F(0.f, 0.f, zPlane), Vector3F(-1,0,0), 0.f, 1e8);
    pleft.rayIntersect(toleft, leftP, tt);
    
    const float leftMost = leftP.x;
    
    const float bottomMost = leftMost * aspectRatio;
    
	glPushMatrix();
	float tmat[16];
    mmvinv.glMatrix(tmat);
    glMultMatrixf(tmat);
    
	glColor3f(1,1,1);
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0); glVertex3f(leftMost,bottomMost, zPlane);
    glTexCoord2f(1, 0); glVertex3f(-leftMost,bottomMost, zPlane);
    glTexCoord2f(1, 1); glVertex3f(-leftMost,-bottomMost, zPlane);
    
    glTexCoord2f(0, 0); glVertex3f(leftMost,bottomMost, zPlane);
    glTexCoord2f(1, 1); glVertex3f(-leftMost,-bottomMost, zPlane);
    glTexCoord2f(0, 1); glVertex3f(leftMost,-bottomMost, zPlane);
    glEnd();
    
    glPopMatrix();
}

void heatherNode::drawBlocks()
{
	if(m_meshes.length() < 1) return;
	unsigned i, j;
	MPointArray points;
	MIntArray vertexList;
	glBegin(GL_TRIANGLES);
	for(i=0; i<m_meshes.length(); i++) {
		MItMeshPolygon iter(m_meshes[i]);
		for(;!iter.isDone();iter.next()) {
			iter.getTriangles(points, vertexList, MSpace::kWorld);
			for(j=0; j<points.length(); j++)
				glVertex3f(points[j].x, points[j].y, points[j].z);
		}
	}
	glEnd();
}

bool heatherNode::isBounded() const
{ 
	return false;
}

void* heatherNode::creator()
{
	return new heatherNode();
}

MStatus heatherNode::initialize()
{ 
	MFnTypedAttribute matAttr;
	MFnNumericAttribute numAttr;
	MStatus			 stat;
	
	amatrix = matAttr.create( "cameraMatrix", "cm", MFnData::kMatrix );
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	addAttribute( amatrix );
	
	anear = numAttr.create( "nearClip", "nc", MFnNumericData::kDouble, 0.1 );
	numAttr.setStorable(true);
	numAttr.setConnectable(true);
	numAttr.setKeyable(true);
	addAttribute( anear );
	
	afar = numAttr.create( "farClip", "fc", MFnNumericData::kDouble, 1000.0 );
	numAttr.setStorable(true);
	numAttr.setConnectable(true);
	numAttr.setKeyable(true);
	addAttribute( afar );
	
	ahapeture = numAttr.create( "horizontalFilmAperture", "hfa", MFnNumericData::kDouble );
	numAttr.setStorable(false);
	numAttr.setConnectable(true);
	addAttribute( ahapeture );
	
	avapeture = numAttr.create( "verticalFilmAperture", "vfa", MFnNumericData::kDouble );
	numAttr.setStorable(false);
	numAttr.setConnectable(true);
	addAttribute( avapeture );
	
	afocallength = numAttr.create( "focalLength", "fl", MFnNumericData::kDouble );
	numAttr.setStorable(false);
	numAttr.setConnectable(true);
	addAttribute( afocallength );
	
	aorthographicwidth = numAttr.create( "orthographicWidth", "ow", MFnNumericData::kDouble );
	numAttr.setStorable(false);
	numAttr.setConnectable(true);
	addAttribute( aorthographicwidth );
	
	aorthographic = numAttr.create( "orthographic", "orh", MFnNumericData::kBoolean );
	numAttr.setStorable(true);
	addAttribute( aorthographic );
	
	adepthImageName = matAttr.create( "depthImage", "dmg", MFnData::kString );
 	matAttr.setStorable(true);
	// stringAttr.setArray(true);
	addAttribute(adepthImageName);
	
	aenableMultiFrames = numAttr.create( "useImageSequence", "uis", MFnNumericData::kBoolean );
	numAttr.setStorable(true);
	addAttribute(aenableMultiFrames);
	
	aframeNumber = numAttr.create( "frameNumber", "fnb", MFnNumericData::kInt );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute(aframeNumber);
	
	aframePadding = numAttr.create( "framePadding", "fpd", MFnNumericData::kInt );
	numAttr.setDefault(0);
	numAttr.setStorable(true);
	addAttribute(aframePadding);
	
	ablockSetName = matAttr.create( "blockSet", "bks", MFnData::kString );
 	matAttr.setStorable(true);
	addAttribute(ablockSetName);
	
	outValue = numAttr.create( "outValue", "ov", MFnNumericData::kFloat );
	numAttr.setStorable(false);
	numAttr.setWritable(false);
	addAttribute(outValue);
	
	attributeAffects(adepthImageName, outValue);
	attributeAffects(aenableMultiFrames, outValue);
	attributeAffects(aframeNumber, outValue);
	attributeAffects(ablockSetName, outValue);
	
	return MS::kSuccess;
}
