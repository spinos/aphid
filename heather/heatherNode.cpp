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
#include <maya/MFnPluginData.h>
#include <SHelper.h>
#include "heather_implement.h"
MTypeId heatherNode::id( 0x7065d6 );

MObject heatherNode::ainimages;
MObject heatherNode::ablockSetName;
MObject heatherNode::acameraName;
MObject heatherNode::outValue;

heatherNode::heatherNode() 
{
    m_numImages = 0;
    m_needLoadImage = 0;
    // m_exr = 0;
	m_framebuffer = 0;
	m_portWidth = 0;
	m_portHeight = 0;
	m_carmeraName = "";
	// m_blockVs = 0;
	// m_blockTriIndices = 0;
	m_combinedColorBuf = new CUDABuffer;
    m_combinedDepthBuf = new CUDABuffer;
    m_hostCombinedColorBuf = new BaseBuffer;
    m_hostCombinedDepthBuf = new BaseBuffer;
}
heatherNode::~heatherNode() 
{
    if(m_framebuffer) delete m_framebuffer;
    delete m_combinedColorBuf;
    delete m_combinedDepthBuf;
    delete m_hostCombinedColorBuf;
    delete m_hostCombinedDepthBuf;
	// if(m_blockVs) delete[] m_blockVs;
	// if(m_blockTriIndices) delete[] m_blockTriIndices;
}

MStatus heatherNode::compute( const MPlug& plug, MDataBlock& block )
{
    if( plug == outValue ) {
        m_numImages = 0;
       
        MArrayDataHandle hArray = block.inputArrayValue(ainimages);
        
        MGlobal::displayInfo("heather compute");
        int numSlots = hArray.elementCount();
        int i;
        for(i=0; i < numSlots; i++) {
            MObject oslot = hArray.inputValue().data();
            MFnPluginData fslot(oslot);
            ExrImgData * dslot = (ExrImgData *)fslot.data();
            if(dslot) {
                ExrImgData::DataDesc * desc = dslot->getDesc();
                addImage(desc);
            }
            hArray.next();
        }
        
        computeCombinedBufs();
        
		MString setname = block.inputValue(ablockSetName).asString();
		m_carmeraName = block.inputValue(acameraName).asString();
		cacheBlocks(setname);
        MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set(0.0);
		
		block.setClean(plug);
		return MS::kSuccess;
    }
	return MS::kUnknownParameter;
}

void heatherNode::addImage(ExrImgData::DataDesc * desc)
{
    if(m_numImages == 32) {
        MGlobal::displayWarning("heather reaches maximum 32 images.");
        return;
    }
    if(!desc->_isValid) return;
    m_images[m_numImages] = desc->_img;
    m_colorBuf[m_numImages] = desc->_colorBuf;
    m_depthBuf[m_numImages] = desc->_depthBuf;
    m_numImages++;
    m_needLoadImage = 1;
}

void heatherNode::computeCombinedBufs()
{
    const unsigned numPix =  m_images[0]->getWidth() * m_images[0]->getHeight();
    m_combinedColorBuf->create(numPix * 4 * 2);
    m_combinedDepthBuf->create(numPix * 4);
    
    void * dstCol = m_combinedColorBuf->bufferOnDevice();
    void * dstDep = m_combinedDepthBuf->bufferOnDevice();
    void * srcCol = m_colorBuf[0]->bufferOnDevice();
    void * srcDep = m_depthBuf[0]->bufferOnDevice();
    
    MGlobal::displayInfo(MString("combine pix ")+numPix);
    CUU::heatherFillImage((CUU::ushort4 *)dstCol, (float *)dstDep, (CUU::ushort4 *)srcCol, (float *)srcDep, numPix);
    
    const unsigned depBufSize = m_depthBuf[0]->bufferSize();
    const unsigned colBufSize = m_colorBuf[0]->bufferSize();
    
    unsigned i = 1;
    for(;i < m_numImages; i++) {
        if(m_depthBuf[i]->bufferSize() != depBufSize) continue;
        if(m_colorBuf[i]->bufferSize() != colBufSize) continue;
        
        MGlobal::displayInfo(MString("combine img ")+i+" "+m_images[i]->getWidth() *  m_images[i]->getHeight());
        
        CUU::heatherMixImage((CUU::ushort4 *)dstCol, 
                             (float *)dstDep, 
                             (CUU::ushort4 *)m_colorBuf[i]->bufferOnDevice(), 
                             (float *)m_depthBuf[i]->bufferOnDevice(), 
                             numPix);
    }
    
    
    m_hostCombinedColorBuf->create(numPix * 4 * 2);
    m_hostCombinedDepthBuf->create(numPix * 4);
    
    m_combinedColorBuf->deviceToHost(m_hostCombinedColorBuf->data(), m_hostCombinedColorBuf->bufferSize());
    m_combinedDepthBuf->deviceToHost(m_hostCombinedDepthBuf->data(), m_hostCombinedDepthBuf->bufferSize());
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
    MDagPath dagCam;
    view.getCamera(dagCam);
    MFnCamera fCam(dagCam);
    if(fCam.isOrtho()) return;
    
    if(m_carmeraName.length() > 3) {
        const MString camName = fCam.name();
        if(camName != m_carmeraName) return;
    }
    
    const double nearPlane = fCam.nearClippingPlane() + .1f;
    
    const double overscan = fCam.overscan();

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
		
		glBindTexture(GL_TEXTURE_2D, m_bgdCImg);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
						
		glBindTexture(GL_TEXTURE_2D, m_colorImg);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
	    glBindTexture(GL_TEXTURE_2D, m_depthImg);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB );
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL );
		m_depth.diagnose(log);
	}	
	
	if(m_numImages < 1) return;
	
    const float imageAspectRatio = m_images[0]->aspectRation();
	
	GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];
    glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
    glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
    glGetIntegerv (GL_VIEWPORT, viewport);
	Matrix44F mmv(mvmatrix);
    Matrix44F mmvinv(mvmatrix); mmvinv.inverse();
    
    const GLint portWidth = viewport[2];
    const GLint portHeight = viewport[3];
	
	if(portWidth != m_portWidth || portHeight != m_portHeight) {
		m_portWidth = portWidth;
		m_portHeight = portHeight;
		
		if(m_framebuffer) delete m_framebuffer;
		m_framebuffer = new GlFramebuffer(portWidth, portHeight);
		// if(m_framebuffer->hasFbo()) MGlobal::displayInfo("fbo created");
		m_clamp.setTextures(m_framebuffer->colorTexture(), m_bgdCImg,
				m_depthImg, 
				m_colorImg);
	}
	
	MFnCamera::FilmFit 	fit = fCam.filmFit();
    int gateWidth, gateHeight;
    float gateSqueezeX = 1.f, gateSqueezeY = 1.f;
    if(fit==MFnCamera::kHorizontalFilmFit) {
        gateWidth = (double)portWidth / overscan;
        gateHeight = gateWidth * imageAspectRatio;
        if(gateHeight < 2) gateHeight = 2;
        if(gateHeight > portHeight) {
            gateSqueezeY = (float)portHeight / (float)gateHeight;
            gateHeight = portHeight;
        }
    }
    else { // assuming it is kVerticalFilmFit
        gateHeight = (double)portHeight / overscan;
        gateWidth = gateHeight / imageAspectRatio;
        if(gateWidth < 2) gateWidth = 2;
        if(gateWidth > portWidth) {
            gateSqueezeX = (float)portWidth / (float)gateWidth;
            gateWidth = portWidth;
        }
    }
    
    const float realRatio = (float)gateHeight/(float)gateWidth;
    
    unsigned char *pixels = new unsigned char[gateWidth * gateHeight * 4];
    
// 8-bit only?
// only GL_DEPTH_COMPONENT works
    glReadPixels((portWidth - gateWidth)/2, (portHeight - gateHeight)/2, gateWidth, gateHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    
    //MGlobal::displayInfo(MString("dep")+pixels[128]);
    
    glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_bgdCImg);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gateWidth, gateHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	
	delete[] pixels;
	
	if(m_needLoadImage) {
    
	    glBindTexture(GL_TEXTURE_2D, m_colorImg);
//#ifdef WIN32
//	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, m_exr->getWidth(), m_exr->getHeight(), 0, GL_RGBA, GL_HALF_FLOAT, m_exr->_pixels);
//#else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, m_images[0]->getWidth(), m_images[0]->getHeight(), 0, GL_RGBA, GL_HALF_FLOAT_ARB, m_hostCombinedColorBuf->data());
//#endif	
	    glBindTexture(GL_TEXTURE_2D, m_depthImg);
	    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB );
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL );
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, m_images[0]->getWidth(), m_images[0]->getHeight(), 0, GL_RED, GL_FLOAT, m_hostCombinedDepthBuf->data());
	    
		m_clamp.setTextures(m_framebuffer->colorTexture(), m_bgdCImg,
				m_depthImg, 
				m_colorImg);
				
		m_needLoadImage = 0;
	}
	
	m_framebuffer->begin();
	m_depth.programBegin();
	glColor3f(1,1,1);
	
	const float gatePortRatioW = (float)gateWidth/(float)portWidth;
	const float gatePortRatioH = (float)gateHeight/(float)portHeight;
	drawBlocks();
	
	m_depth.programEnd();
	m_framebuffer->end();
	
	glDisable(GL_DEPTH_TEST);
	
	m_clamp.setClippings(0.1, fCam.farClippingPlane());

	glPushAttrib(GL_ALL_ATTRIB_BITS);
    m_clamp.programBegin();
    
	drawBackPlane(nearPlane, projmatrix, mmvinv, realRatio, overscan, gatePortRatioW, gatePortRatioH, fit==MFnCamera::kHorizontalFilmFit, gateSqueezeX, gateSqueezeY);
    
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

void heatherNode::drawBackPlane(double farPlane, const GLdouble * mproj, const Matrix44F & mmvinv, const float & aspectRatio, const double & overscan,
	                   const float & gatePortRatioW, const float & gatePortRatioH, char isHorizontalFit,
	                   float gateSqueezeX, float gateSqueezeY)
{
	float tt;
    Vector3F leftP;
    
    double dt = -farPlane;

    double leftMost, bottomMost;
    
    if(isHorizontalFit) {
        Plane pleft(mproj[3] + mproj[0], 
               mproj[7] + mproj[4],
               mproj[11] + mproj[8], 
               mproj[15] + mproj[12]);
    
        Ray toleft(Vector3F(0.f, 0.f, dt), Vector3F(-1,0,0), 0.f, 1e32);
        pleft.rayIntersect(toleft, leftP, tt);
        
        leftMost = leftP.x / overscan;
        bottomMost = leftMost * aspectRatio;
    }
    else {
        Plane pbottom(mproj[3] + mproj[1], 
               mproj[7] + mproj[5],
               mproj[11] + mproj[9], 
               mproj[15] + mproj[13]);
    
        Ray tobottom(Vector3F(0.f, 0.f, dt), Vector3F(0,-1,0), 0.f, 1e32);
        pbottom.rayIntersect(tobottom, leftP, tt);
        
        bottomMost = leftP.y / overscan;
        leftMost = bottomMost / aspectRatio;
    }
    
    // MGlobal::displayInfo(MString("corners ")+leftMost+" "+bottomMost+" "+dt);
    
	glPushMatrix();
	float tmat[16];
    mmvinv.glMatrix(tmat);
    glMultMatrixf(tmat);
    
    const float portL = (1.f - gatePortRatioW) * .5f;
    const float portR = portL + gatePortRatioW;
    const float portB = (1.f - gatePortRatioH) * .5f;
    const float portT = portB + gatePortRatioH;
    
    const float gateL = (1.f - gateSqueezeX) * .5f;
    const float gateR = gateL + gateSqueezeX;
    const float gateB = (1.f - gateSqueezeY) * .5f;
    const float gateT = gateB + gateSqueezeY;
    
	glColor3f(1,1,1);
    glBegin(GL_TRIANGLES);
    glMultiTexCoord2f(GL_TEXTURE0, gateL, gateB); 
    glMultiTexCoord2f(GL_TEXTURE1, portL, portB); 
    glMultiTexCoord2f(GL_TEXTURE2, gateL, gateT); 
    glVertex3d(leftMost,bottomMost, dt);
    
    glMultiTexCoord2f(GL_TEXTURE0, gateR, gateB);
    glMultiTexCoord2f(GL_TEXTURE1, portR, portB); 
    glMultiTexCoord2f(GL_TEXTURE2, gateR, gateT); 
    glVertex3d(-leftMost,bottomMost, dt);
    
    glMultiTexCoord2f(GL_TEXTURE0, gateR, gateT);
    glMultiTexCoord2f(GL_TEXTURE1, portR, portT); 
    glMultiTexCoord2f(GL_TEXTURE2, gateR, gateB); 
    glVertex3d(-leftMost,-bottomMost, dt);
    
    glMultiTexCoord2f(GL_TEXTURE0, gateL, gateB); 
    glMultiTexCoord2f(GL_TEXTURE1, portL, portB); 
    glMultiTexCoord2f(GL_TEXTURE2, gateL, gateT); 
    glVertex3d(leftMost,bottomMost, dt);
    
    glMultiTexCoord2f(GL_TEXTURE0, gateR, gateT); 
    glMultiTexCoord2f(GL_TEXTURE1, portR, portT); 
    glMultiTexCoord2f(GL_TEXTURE2, gateR, gateB); 
    glVertex3f(-leftMost,-bottomMost, dt);
    
    glMultiTexCoord2f(GL_TEXTURE0, gateL, gateT); 
    glMultiTexCoord2f(GL_TEXTURE1, portL, portT); 
    glMultiTexCoord2f(GL_TEXTURE2, gateL, gateB); 
    glVertex3d(leftMost,-bottomMost, dt);
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
	
	ainimages = matAttr.create("inImage", "iim", MFnData::kPlugin);
	matAttr.setStorable(false);
	matAttr.setConnectable(true);
	matAttr.setArray(true);
	addAttribute(ainimages);
	
	acameraName = matAttr.create( "lookCameraName", "lcm", MFnData::kString );
 	matAttr.setStorable(true);
	addAttribute(acameraName);
	
	ablockSetName = matAttr.create( "blockSet", "bks", MFnData::kString );
 	matAttr.setStorable(true);
	addAttribute(ablockSetName);
	
	outValue = numAttr.create( "outValue", "ov", MFnNumericData::kFloat );
	numAttr.setStorable(false);
	numAttr.setWritable(false);
	addAttribute(outValue);
	
	attributeAffects(ainimages, outValue);
	attributeAffects(ablockSetName, outValue);
	attributeAffects(acameraName, outValue);
	
	return MS::kSuccess;
}
