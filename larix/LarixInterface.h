/*
 *  LarixInterface.h
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <HesperisInterface.h>
class KdTreeDrawer;
class APointCloud;
class ATetrahedronMesh;
class LarixWorld;
class AdaptiveField;
class BaseBuffer;

class LarixInterface : public HesperisInterface {
public:
	LarixInterface();
	virtual ~LarixInterface();
	
	bool createWorld(LarixWorld * world);
	void drawWorld(LarixWorld * world, KdTreeDrawer * drawer);
protected:
	static APointCloud * ConvertTetrahedrons(ATetrahedronMesh * mesh);
private:
    void drawField(AdaptiveField * field, 
                          const std::string & channelName,
                          KdTreeDrawer * drawer);
	void drawGrid(AdaptiveField * field,
					KdTreeDrawer * drawer);
    void buildCells(AdaptiveField * fld);
private:
// cell center and size
    BaseBuffer * m_cells;
};