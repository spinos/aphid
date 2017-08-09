/*
 *  file.h
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_FILE_H
#define GAR_FILE_H

namespace gar {
    
#define NUM_FILE_PIECES 2

static const char * FileTypeNames[NUM_FILE_PIECES] = {
"unknown",
"Import Geometry"
};

static const char * FileTypeImages[NUM_FILE_PIECES] = {
"unknown",
":/icons/unknown.png"
};

static const char * FileTypeIcons[NUM_FILE_PIECES] = {
":/icons/unknown.png",
":/icons/import_geom.png"
};

static const char * FileTypeDescs[NUM_FILE_PIECES] = {
"unknown",
"import geometry from .hes file \n 1 deviations\n height unknown unit"
};

static inline int ToFileType(int x) {
	return x - 128;
}

static const int FileInPortRange[NUM_FILE_PIECES][2] = {
{0,0},
{0,0},
};

static const char * FileInPortRangeNames[2] = {
"",
""
};

static const int FileOutPortRange[NUM_FILE_PIECES][2] = {
{0,0},
{0,1},
};

static const char * FileOutPortRangeNames[2] = {
"outStem",
""
};

static const int FileGeomDeviations[NUM_FILE_PIECES] = {
0,
1,
};

}
#endif
