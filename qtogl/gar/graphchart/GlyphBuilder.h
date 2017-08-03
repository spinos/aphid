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
	
};
#endif