/*
 *  BccInterface.h
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <HesperisInterface.h>
class KdTreeDrawer;
class AGenericMesh;
class ATetrahedronMesh;
class BccWorld;
class AdaptiveField;
class BaseBuffer;

class BccInterface : public HesperisInterface {
public:
	BccInterface();
	virtual ~BccInterface();
	
	bool createWorld(BccWorld * world);
	void drawWorld(BccWorld * world, KdTreeDrawer * drawer);
	bool saveWorld(BccWorld * world);
	
	bool loadTriangleGeometry(BccWorld * world, const std::string & filename);
	bool loadCurveGeometry(BccWorld * world, const std::string & filename);
	bool loadPatchGeometry(BccWorld * world, const std::string & filename);
protected:
	
private:
    void drawTetrahedronMesh(ATetrahedronMesh * m, KdTreeDrawer * drawer);
	void drawGeometry(GeometryArray * geos, KdTreeDrawer * drawer);
	void drawAnchors(AGenericMesh * mesh, KdTreeDrawer * drawer,
						float drawSize);
private:
// cell center and size
    BaseBuffer * m_cells;
};