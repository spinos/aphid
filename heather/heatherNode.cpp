///:
// heatherNode.cpp
// Zhang Jian
// 07/12/05

#include "heatherNode.h"

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
MObject heatherNode::aframeNumber;
MObject heatherNode::outValue;

heatherNode::heatherNode() 
{
    m_needLoadImage = 0;
    m_exr = 0;
}
heatherNode::~heatherNode() 
{
    if(m_exr) delete m_exr;
}

MStatus heatherNode::compute( const MPlug& plug, MDataBlock& block )
{ 
    if( plug == outValue ) {
        MString filename =  block.inputValue(adepthImageName).asString();
        int frame = block.inputValue(aframeNumber).asInt();
        preLoadImage(filename.asChar(), frame);
        MDataHandle outputHandle = block.outputValue( outValue );
		outputHandle.set(0.0);
    }
	return MS::kSuccess;
}

void heatherNode::preLoadImage(const char * name, int frame)
{
    std::string fileName(name);
    if(fileName.size() < 3) return;
    
	if(m_exr)
		delete m_exr;
	
    m_exr = new ZEXRImage(fileName.c_str());
    if(!m_exr->isOpened()) return;

    MGlobal::displayInfo(MString("loading image ")+ name+" at frame "+frame);
	
    if(!m_exr->isRGBAZ()) {
        MGlobal::displayWarning("image is not RGBAZ format.");
        
        return;
    }
    
    m_needLoadImage = 1;
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
	
#ifdef WIN32
    if(!isDiagnosed()) {
        MGlobal::displayInfo("init glext on win32");
		gExtensionInit();
		std::string log;
		diagnose(log);
		MGlobal::displayInfo(MString("glsl diagnose log: ") + log.c_str());
	}
#endif	
	
	GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];
    glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
    glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
    glGetIntegerv (GL_VIEWPORT, viewport);
	Matrix44F mmv(mvmatrix);
    Matrix44F mmvinv(mvmatrix); mmvinv.inverse();
    Matrix44F mproj(projmatrix);
    
    const GLint width = viewport[2];
    const GLint height = viewport[3];
    
    GLfloat *pixels = new GLfloat[width * height];
    
    MGlobal::displayInfo(MString("port wh")+width+" "+height);
    
    //glPixelStorei(GL_UNPACK_ROW_LENGTH, viewport[2]);
    //int rowSkip = 0;
    //int pixelSkip = 0;
    //glPixelStorei(GL_UNPACK_SKIP_PIXELS, pixelSkip);
    //glPixelStorei(GL_UNPACK_SKIP_ROWS, rowSkip);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT,GL_FLOAT, pixels);
    
    // for(int i =0; i < 1024; i++) pixels[i]= ((float)(rand() % 99)) / 99.f;
    MGlobal::displayInfo(MString("read pix")+width+" "+height);
    
	glGenTextures(1, &m_depthImg);
	glBindTexture(GL_TEXTURE_2D, m_depthImg);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL );
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_RED, GL_FLOAT, pixels);
	
	MGlobal::displayInfo(MString("tex pix")+width+" "+height);
	delete[] pixels;
	
	float tt;
    Vector3F leftP;
    
    Plane pfar(mproj.M(0,3) - mproj.M(0,2), 
               mproj.M(1,3) - mproj.M(1,2),
               mproj.M(2,3) - mproj.M(2,2), 
               mproj.M(3,3) - mproj.M(3,2));
    // MGlobal::displayInfo(MString("proj")+mproj.str().c_str());
    Ray toFar(Vector3F(0,0,0), Vector3F(0,0,-1), 0.f, 1e8);
    
    pfar.rayIntersect(toFar, leftP, tt);
    MGlobal::displayInfo(MString("far")+leftP.str().c_str());

    Plane pleft(mproj.M(0,3) + mproj.M(0,0), 
               mproj.M(1,3) + mproj.M(1,0),
               mproj.M(2,3) + mproj.M(2,0), 
               mproj.M(3,3) + mproj.M(3,0));
    
    const float zPlane = leftP.z * .999f;
    Ray toleft(Vector3F(0.f, 0.f, zPlane), Vector3F(-1,0,0), 0.f, 1e8);
    pleft.rayIntersect(toleft, leftP, tt);
    // qDebug()<<"left"<<leftP.str().c_str();
    const float leftMost = leftP.x;
    
    Plane pbottom(mproj.M(0,3) + mproj.M(0,1), 
               mproj.M(1,3) + mproj.M(1,1),
               mproj.M(2,3) + mproj.M(2,1), 
               mproj.M(3,3) + mproj.M(3,1));
    
    Ray tobottom(Vector3F(0.f, 0.f, zPlane), Vector3F(0,-1,0), 0.f, 1e8);
    pbottom.rayIntersect(tobottom, leftP, tt);
    // qDebug()<<"bottom"<<leftP.str().c_str();
    const float bottomMost = leftP.y;
	
    programBegin();
    
	glPushMatrix();
	float tmat[16];
    mmvinv.glMatrix(tmat);
    glMultMatrixf(tmat);
    
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0); glVertex3f(leftMost,bottomMost, zPlane);
    glTexCoord2f(1, 0); glVertex3f(-leftMost,bottomMost, zPlane);
    glTexCoord2f(1, 1); glVertex3f(-leftMost,-bottomMost, zPlane);
    glEnd();
    
    glPopMatrix();
    
	programEnd();
	
	glDeleteTextures(1, &m_depthImg);
	glEnable(GL_DEPTH_TEST);
	view.endGL();
	
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
	
	aframeNumber = numAttr.create( "frameNumber", "fnb", MFnNumericData::kInt );
	numAttr.setStorable(true);
	numAttr.setKeyable(true);
	addAttribute(aframeNumber);
	
	outValue = numAttr.create( "outValue", "ov", MFnNumericData::kFloat );
	numAttr.setStorable(false);
	numAttr.setWritable(false);
	addAttribute(outValue);
	
	attributeAffects(adepthImageName, outValue);
	attributeAffects(aframeNumber, outValue);
	
	return MS::kSuccess;
}

void heatherNode::updateShaderParameters() const
{
    glUniform1iARB(glGetUniformLocationARB(program_object, "color_texture"), 0);
    glActiveTexture(GL_TEXTURE0);
    glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, m_depthImg);
}

const char* heatherNode::vertexProgramSource() const
{
	return "void main()"
"{"
"		gl_Position = ftransform();"
"		gl_FrontColor = gl_Color;"
"gl_TexCoord[0] = gl_MultiTexCoord0;"
"}";
}

const char* heatherNode::fragmentProgramSource() const
{
	return "uniform sampler2D color_texture;"
"void main()"
"{"
"float d = texture2D(color_texture, gl_TexCoord[0].xy).r;"
"gl_FragColor = vec4(d, d, d,1.0);"
"}";
}



