/*
 *  DrawVegetation.h
 *  
 *
 *  Created by jian zhang on 4/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_DRAW_VEGETATION_H
#define GAR_DRAW_VEGETATION_H

namespace aphid {
class ATriangleMesh;
}

class PlantPiece;
class VegetationPatch;

class DrawVegetation {

public:
	DrawVegetation();
	virtual ~DrawVegetation();
	
	void drawNaive(const VegetationPatch * vgp);
	void drawPlantPatch(const VegetationPatch * vgp);
	void drawPointPatch(const VegetationPatch * vgp);
	void drawPlant(const PlantPiece * pl);
	
protected:

private:
	void drawPiece(const PlantPiece * pl);
	void drawMesh(const aphid::ATriangleMesh * geo);
	
};

#endif