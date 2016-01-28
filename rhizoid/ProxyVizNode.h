#ifndef PROXYVIZNODE_H
#define PROXYVIZNODE_H

/*
 *  proxyVizNode.h
 *  proxyPaint
 *
 *  Created by jian zhang on 3/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/M3dView.h>
#include <maya/MMatrixArray.h>
#include <maya/MIntArray.h>
#include <maya/MFloatArray.h>
#include <maya/MFnMesh.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <Matrix44F.h>
#include <Vector3F.h>
#include "depthCut.h"

class MItMeshPolygon;

class ProxyViz : public MPxLocatorNode
{
	double m_materializePercentage;
	
public:
	ProxyViz();
	virtual ~ProxyViz(); 

    virtual MStatus   		compute( const MPlug& plug, MDataBlock& data );

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );

	virtual bool            isBounded() const;
	virtual MBoundingBox    boundingBox() const; 

	static  void *          creator();
	static  MStatus         initialize();

	static  MObject         abboxminx;
	static  MObject         abboxminy;
	static  MObject         abboxminz;
	static  MObject         abboxmaxx;
	static  MObject         abboxmaxy;
	static  MObject         abboxmaxz;
	static MObject outPositionPP;
	static MObject outScalePP;
	static MObject outRotationPP;
	static MObject acachename;
	static MObject adumpname;
	static MObject acameraspace;
	static MObject alodgatehigh;
	static MObject alodgatelow;
	static MObject ahapeture;
	static MObject avapeture;
	static MObject afocallength;
	static MObject axmultiplier;
	static MObject aymultiplier;
	static MObject azmultiplier;
	static MObject agroupcount;
	static MObject astarttime;
	static MObject ainstanceId;
	static MObject aenablecull;
	static MObject ainmesh;
	static MObject aconvertPercentage;
	static MObject astandinNames;
	static MObject outValue;
	
	void addABox(const MMatrix & m);
	char isBoxInView(const MPoint &pos, float threshold, short xmin, short ymin, short xmax, short ymax);
	void selectBoxesInView(short xmin, short ymin, short xmax, short ymax, MGlobal::ListAdjustment selectionMode);
	void removeBoxesInView(short xmin, short ymin, short xmax, short ymax, const float & threshold);
	void adjustSize(short x, short y, float magnitude);
	void adjustPosition(short x0, short y0, short x1, short y1, float clipNear, float clipFar, Matrix44F &mat, MFnMesh &mesh);
	void smoothPosition(short x0, short y0, short x1, short y1, float clipNear, float clipFar, Matrix44F &mat, MFnMesh &mesh);
	void adjustRotation(short x, short y, float magnitude, short axis, float noise = 0.f);
	void adjustLocation(short start_x, short start_y, short last_x, short last_y, float clipNear, float clipFar, Matrix44F & mat, short axis, float noise = 0.f);
	void pressToSave();
	void pressToLoad();
	void removeAllBoxes();
	void setCullMesh(MDagPath mesh);
	void snapByIntersection(MFnMesh &mesh);
	
	unsigned getNumActiveBoxes() const;
	MMatrix getActiveBox(unsigned idx) const;
	int getActiveIndex(unsigned idx) const;
	void setActiveBox(unsigned idx, const MMatrix & mat);
	
	const MMatrixArray & spaces() const;
	const MMatrixArray spaces(int groupCount, int groupId, MIntArray & ppNums) const;
	const MMatrix & worldSpace() const;
public: 
	static	MTypeId		id;
	
private:
	char hasDisplayMesh() const;
	std::string replaceEnvVar(const MString & filename) const;
	
	MMatrix _worldSpace, _worldInverseSpace;
	MMatrixArray _spaces;
	MFloatArray _details;
	MIntArray _randNums;
	MIntArray _activeIndices;
	M3dView _viewport;
	Vector3F _bbmin, _bbmax;
	char *fVisibleTag;
	char _firstLoad, fHasView;
	void drawSelected(float mScale[16]);
	void drawSolidMesh(MItMeshPolygon & iter);
	void drawWireMesh(MItMeshPolygon & iter);
	void draw_solid_box() const;
	void draw_a_box() const;
	void draw_coordsys() const;
	void drawViewFrustum(const MMatrix & cameraSpace, const float & h_fov, const float & aspectRatio);
	void loadCache(const char* filename);
	void saveCache(const char* filename);
	void bakePass(const char* filename, const MVectorArray & position, const MVectorArray & scale, const MVectorArray & rotation);
	void calculateLOD(const MMatrix & cameraInv, const float & h_fov, const float & aspectRatio, const float & detail, const int & enableViewFrustumCulling);
	void updateWorldSpace();
	MMatrix localizeSpace(const MMatrix & s) const;
	MMatrix worldizeSpace(const MMatrix & s) const;
	void useActiveView();
	DepthCut * fCuller;
	MObject fDisplayMesh;
	
};
#endif        //  #ifndef PROXYVIZNODE_H
