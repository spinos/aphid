/*
 *  GlyphBuilder.h
 *  garden
 *
 *  add ports
 *
 *  Created by jian zhang on 4/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_GLYPH_BUILDER_H
#define GAR_GLYPH_BUILDER_H

class PieceAttrib;
class GardenGlyph;

class GlyphBuilder {
	
public:
	GlyphBuilder();
	virtual ~GlyphBuilder();
	
	void build(GardenGlyph * dst,
			const int & gtyp,
			const int & ggrp);
	
protected:

private:
	void buildGround(GardenGlyph * dst,
			const int & gtyp);
	void buildGrass(GardenGlyph * dst,
			const int & gtyp);
	void buildFile(GardenGlyph * dst,
			const int & gtyp);
	void buildSprite(GardenGlyph * dst,
			const int & gtyp);
	void buildVariant(GardenGlyph * dst,
			const int & gtyp);
	void buildStem(GardenGlyph * dst,
			const int & gtyp);
	void buildTwig(GardenGlyph * dst,
			const int & gtyp);
			
	PieceAttrib* buildAttrib(const int & gtyp,
			const int & ggrp);
			
	PieceAttrib* buildGroundAttrib(const int & gtyp);
	PieceAttrib* buildFileAttrib(const int & gtyp);
	PieceAttrib* buildSpriteAttrib(const int & gtyp);
	PieceAttrib* buildVariantAttrib(const int & gtyp);
	PieceAttrib* buildGrassAttrib(const int & gtyp);
	PieceAttrib* buildStemAttrib(const int & gtyp);
	PieceAttrib* buildTwigAttrib(const int & gtyp);
	
};
#endif