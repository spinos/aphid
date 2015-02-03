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
 

class heatherNode : public MPxLocatorNode
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
	MMatrix modelViewMatrix;
	view.modelViewMatrix (modelViewMatrix );
	
	MPoint p0(-1.f, 0.f, 0.f); p0 *= modelViewMatrix;
				
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex3f(p0.x, p0.y, p0.z);
    glTexCoord2f(1, 0); glVertex2f(1, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();
	
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
