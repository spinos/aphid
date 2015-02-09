///:
// heatherNode.cpp
// Zhang Jian
// 07/12/05

#include <math.h>
#include <maya/MIOStream.h>
#include <maya/MPxLocatorNode.h> 
#include <maya/MString.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MVector.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MColor.h>
#include <maya/M3dView.h>
#include <maya/MFnPlugin.h>
#include <maya/MDistance.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MMatrix.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MPoint.h>
#include <maya/MString.h>
#include <maya/MDagPath.h>
#include <maya/MSelectionList.h>
#include <maya/MFnCamera.h>
#include <maya/MGlobal.h>
#include <Plane.h>
#include <glslBase.h>
class heatherNode : public MPxLocatorNode, public GLSLBase
{
public:
	heatherNode();
	virtual ~heatherNode(); 

    virtual MStatus   		compute( const MPlug& plug, MDataBlock& data );

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );

	virtual bool            isBounded() const;

	static  void *          creator();
	static  MStatus         initialize();

	static  MObject         amatrix;
	static  MObject         anear;
	static  MObject         afar;
	static	MObject		ahapeture;
	static	MObject		avapeture;
	static	MObject		afocallength;
	static	MObject		aorthographic;
	static	MObject		aorthographicwidth;

public: 
	static	MTypeId		id;
	
protected:
    virtual const char* vertexProgramSource() const;
	virtual const char* fragmentProgramSource() const;
	virtual void updateShaderParameters() const;
private:
    GLuint m_depthImg;
};

MTypeId heatherNode::id( 0x0002919 );

MObject heatherNode::amatrix;
MObject heatherNode::anear;
MObject heatherNode::afar;
MObject heatherNode::ahapeture;
MObject heatherNode::avapeture;
MObject heatherNode::afocallength;
MObject heatherNode::aorthographic;
MObject heatherNode::aorthographicwidth;

heatherNode::heatherNode() {}
heatherNode::~heatherNode() {}

MStatus heatherNode::compute( const MPlug& /*plug*/, MDataBlock& /*data*/ )
{ 
	return MS::kUnknownParameter;
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
    
    glPixelStorei(GL_UNPACK_ROW_LENGTH, viewport[2]);
    int rowSkip = 0;
    int pixelSkip = 0;
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, pixelSkip);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, rowSkip);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT,GL_FLOAT, pixels);
    
    // for(int i =0; i < 1024; i++) pixels[i]= ((float)(rand() % 99)) / 99.f;
    MGlobal::displayInfo(MString("read pix")+width+" "+height);
    
	glGenTextures(1, &m_depthImg);
	glBindTexture(GL_TEXTURE_2D, m_depthImg);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_FLOAT, GL_FLOAT, pixels);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	
	MGlobal::displayInfo(MString("tex pix")+width+" "+height);
	
	float tt;
    Vector3F leftP;
    
    Plane pnear(mproj.M(0,2), 
               mproj.M(1,2),
               mproj.M(2,2), 
               mproj.M(3,2));
    
    Ray toNear(Vector3F(0,0,0), Vector3F(0,0,1), 0.f, 1000.f);
    
    pnear.rayIntersect(toNear, leftP, tt);
    // qDebug()<<"near"<<leftP.str().c_str();

    Plane pleft(mproj.M(0,3) + mproj.M(0,0), 
               mproj.M(1,3) + mproj.M(1,0),
               mproj.M(2,3) + mproj.M(2,0), 
               mproj.M(3,3) + mproj.M(3,0));
    
    const float zPlane = leftP.z * 1.01f;
    Ray toleft(Vector3F(0.f, 0.f, zPlane), Vector3F(-1,0,0), 0.f, 1000.f );
    pleft.rayIntersect(toleft, leftP, tt);
    // qDebug()<<"left"<<leftP.str().c_str();
    const float leftMost = leftP.x;
    
    Plane pbottom(mproj.M(0,3) + mproj.M(0,1), 
               mproj.M(1,3) + mproj.M(1,1),
               mproj.M(2,3) + mproj.M(2,1), 
               mproj.M(3,3) + mproj.M(3,1));
    
    Ray tobottom(Vector3F(0.f, 0.f, zPlane), Vector3F(0,-1,0), 0.f, 1000.f );
    pbottom.rayIntersect(tobottom, leftP, tt);
    // qDebug()<<"bottom"<<leftP.str().c_str();
    const float bottomMost = leftP.y;
	
    programBegin();
    
	glPushMatrix();
	float tmat[16];
    mmvinv.glMatrix(tmat);
    glMultMatrixf(tmat);
    
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0); glVertex3f(leftMost,bottomMost, zPlane);
    glTexCoord2f(1, 0); glVertex3f(-leftMost,bottomMost, zPlane);
    glTexCoord2f(1, 1); glVertex3f(-leftMost,-bottomMost, zPlane);
    glEnd();
    
    glPopMatrix();
    
	programEnd();
	delete[] pixels;
	glDeleteTextures(1, &m_depthImg);
	
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
	
	return MS::kSuccess;
}

void heatherNode::updateShaderParameters() const
{
    glActiveTexture(GL_TEXTURE0);
    //int texture_location = glGetUniformLocationARB(program_object, "color_texture");
    //glUniform1iARB(texture_location, 0);
    //glBindTexture(GL_TEXTURE_2D, m_depthImg);
}

const char* heatherNode::vertexProgramSource() const
{
	return "varying vec3 PCAM;"
"void main()"
"{"
"		gl_Position = ftransform();"
"		gl_FrontColor = gl_Color;"
"gl_TexCoord[0] = gl_MultiTexCoord0;"
"	PCAM = vec3 (gl_ModelViewMatrix * gl_Vertex);"
"}";
}

const char* heatherNode::fragmentProgramSource() const
{
	return "varying vec3 PCAM;"
"uniform sampler2D color_texture;"
"void main()"
"{"
"	float d = -PCAM.z; "
"		gl_FragColor = vec4 (vec3(gl_FragCoord.z / 10.f), 1.0);"
"gl_FragColor = texture2D(color_texture, gl_TexCoord[0].st);"
"}";
}

MStatus initializePlugin( MObject obj )
{ 
	MStatus   stat;
	MFnPlugin plugin( obj, "ZHANG JIAN - Free Downlaod", "3.0", "Any");

	stat = plugin.registerNode( "heatherNode", heatherNode::id, 
						 &heatherNode::creator, &heatherNode::initialize,
						 MPxNode::kLocatorNode );
	if (!stat) {
		stat.perror("registerNode");
		return stat;
	}

	// MGlobal::executeCommand ( "source cameraFrustumMenus.mel;cameraFrustumCreateMenus" );

	return stat;
}

MStatus uninitializePlugin( MObject obj)
{
	MStatus   stat;
	MFnPlugin plugin( obj );

	stat = plugin.deregisterNode( heatherNode::id );
	if (!stat) {
		stat.perror("deregisterNode");
		return stat;
	}

	// MGlobal::executeCommand ( "cameraFrustumRemoveMenus" );

	return stat;
}

