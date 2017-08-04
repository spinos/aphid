/*
 *  PieceAttrib.h
 *  collection of attribs
 *
 *  Created by jian zhang on 8/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_PIECE_ATTRIB_H
#define GAR_PIECE_ATTRIB_H

#include <string>

class PieceAttrib {

	std::string m_glyphName;
	
public:
	enum AttribType {
		tUnknown = 0,
		tString = 1,
		tStringFileName = 2,
		tFloat = 3
	};
	
	PieceAttrib();
	virtual ~PieceAttrib();
	
	const std::string& glyphName() const;
	
protected:

private:
};

#endif
