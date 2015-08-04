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

class LarixInterface : public HesperisInterface {
public:
	LarixInterface();
	virtual ~LarixInterface();
	
	static bool CreateWorld(LarixWorld * world);
	static void DrawWorld(LarixWorld * world, KdTreeDrawer * drawer);
protected:
	static APointCloud * ConvertTetrahedrons(ATetrahedronMesh * mesh);
private:
    static void DrawField(AdaptiveField * field, 
                          const std::string & channelName,
                          KdTreeDrawer * drawer);
};