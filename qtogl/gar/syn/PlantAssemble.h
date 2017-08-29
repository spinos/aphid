/*
 *  PlantAssemble.h
 *  
 *
 *  Created by jian zhang on 8/29/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_PLANT_ASSEMBLE_H
#define GAR_PLANT_ASSEMBLE_H

class Vegetation;
class VegetationPatch;

namespace aphid {
class GlyphPort;
class ATriangleMesh;
}

class GardenGlyph;
class PlantPiece;

namespace gar {

class PlantAssemble {

	Vegetation * m_vege;
	
public:
	PlantAssemble(Vegetation * vege);
	
	void genSinglePlant();
	void genMultiPlant();
	
protected:
	virtual GardenGlyph* getGround();
/// smallest exclr of piece connected to ground
	float getMinExclR(GardenGlyph * gnd) const;
	
	Vegetation* vegetationR();

	void growOnGround(VegetationPatch * vpatch, GardenGlyph * gnd);
	
private:
	void assemblePlant(PlantPiece * pl, GardenGlyph * gl);
	void addPiece(PlantPiece * pl, const aphid::GlyphPort * pt);
	void addSinglePiece(PlantPiece * pl, GardenGlyph * gl);
	void addTwigPiece(PlantPiece * pl, GardenGlyph * gl);
	void addBranchPiece(PlantPiece * pl, GardenGlyph * gl);
	
};

}

#endif

