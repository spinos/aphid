/*
 *  frustumBoxCmd.h
 *  frustumBox
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "frustumbox_common.h"
#include <maya/MSyntax.h>
#include <maya/MPxCommand.h> 
#include <maya/MArgDatabase.h>
#include <vector>
#include <AllMath.h>
#include <sdb/VectorArray.h>
#include <ConvexShape.h>

class FrustumBoxCmd : public MPxCommand 
{
public:
	FrustumBoxCmd();
	virtual ~FrustumBoxCmd();

	virtual MStatus		doIt(const MArgList &argList);
	static MSyntax newSyntax();
	static  void* creator();
	MStatus			redoIt();

protected:
	virtual MStatus			parseArgs(const MArgList &argList);
	MStatus printHelp();
	void collide(std::vector<int> & visibilities,
                 const MDagPathArray & paths,
                 MObject & camera) const;
    void worldFrustum(aphid::Vector3F * corners, MObject & camera) const;
/// mesh only
    void worldOrientedBox(aphid::Vector3F * corners, const MDagPath & path) const;
    void worldBBox(aphid::Vector3F * corners, const MDagPath & path) const;
    bool isAllVisible(const std::vector<int> & visibilities) const;
    void listObjects(MStringArray & dst,
                     const MDagPathArray & paths,
                     const std::vector<int> & visibilities,
                     bool visible);
					 
	void getMeshTris(aphid::sdb::VectorArray<aphid::cvx::Triangle> & tris,
					aphid::BoundingBox & bbox,
					const MDagPath & meshPath,
					const MDagPath & tansformPath) const;
								
private:
	enum WorkMode {
		WHelp = 0,
		WAction = 1
	};
	WorkMode m_mode;
	MString m_cameraName;
	double m_cameraScale;
    int m_startTime, m_endTime;
    bool m_doListVisible;
};