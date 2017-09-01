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
class PieceAttrib;

namespace gar {

class SynthesisGroup;

class PlantAssemble {

	Vegetation * m_vege;
/// smallest exclr of piece connected to ground
	float m_exclR;
	
public:
	PlantAssemble(Vegetation * vege);
	
	void genSinglePlant();
	void genMultiPlant();
	
protected:
	virtual GardenGlyph* getGround();

	void estimateExclusionRadius(GardenGlyph * gnd);
	
	Vegetation* vegetationR();

	void growOnGround(VegetationPatch * vpatch, GardenGlyph * gnd);
/// make sure ground has input connection
	GardenGlyph* checkGroundConnection(GardenGlyph* gnd);
	
private:
	void assemblePlant(PlantPiece * pl, GardenGlyph * gl);
	void addPiece(PlantPiece * pl, const aphid::GlyphPort * pt);
	void addSinglePiece(PlantPiece * pl, GardenGlyph * gl);
	void addTwigPiece(PlantPiece * pl, GardenGlyph * gl);
	void addBranchPiece(PlantPiece * pl, GardenGlyph * gl);
	void addSynthesisPiece(PlantPiece * pl, PieceAttrib* attr, 
							SynthesisGroup* syng);
	
};

}

#endif

