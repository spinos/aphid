/*
 *  StickyLocator.h
 *  manuka
 *
 *  Created by jian zhang on 1/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <maya/MFnMesh.h>
#include <maya/MPxLocatorNode.h> 

namespace lfr {
template <typename T> class DenseMatrix;
template <typename T> class DenseVector;
template <typename T> class SvdSolver;
}

class CircleCurve;

class StickyLocator : public MPxLocatorNode
{
	MPoint m_origin;
	float m_refScale;
	CircleCurve * m_circle;
	lfr::DenseMatrix<float> *m_P;
	lfr::DenseMatrix<float> *m_Q;
	lfr::DenseMatrix<float> *m_S;
	lfr::DenseMatrix<float> *m_Vd;
	lfr::DenseMatrix<float> *m_Ri;
	lfr::DenseMatrix<float> *m_scad;
	lfr::SvdSolver<float> *m_svdSolver;
	
public:
	StickyLocator();
	virtual ~StickyLocator(); 

    virtual MStatus   		compute(const MPlug& plug, MDataBlock &data);

	virtual void            draw(M3dView &view, const MDagPath &path, 
								 M3dView::DisplayStyle style,
								 M3dView::DisplayStatus status);

	virtual bool            isBounded() const;
	virtual MBoundingBox    boundingBox() const; 

	static  void *          creator();
	static  MStatus         initialize();

	static MObject         size;
	static MObject aMoveVX;
	static MObject aMoveVY;
	static MObject aMoveVZ;
	static MObject aMoveV;
	static MObject ainmesh;
	static MObject avertexId;
	static  	MObject 	ainrefi;
	static  	MObject 	ainrefd;
	static MObject avertexSpace;
	
public: 
	static	MTypeId		id;
	
private:
	void drawCircle() const;
	void buildRefScale(const MVectorArray & diffs);
	void updateRotation(const MFnMesh & fmesh, 
					const MIntArray & indices, 
					const MVectorArray & diffs);
};
