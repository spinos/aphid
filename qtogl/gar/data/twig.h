/*
 *  twig.h
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_TWIG_H
#define GAR_TWIG_H

namespace gar {
    
#define NUM_TWIG_PIECES 2

static const char * TwigTypeNames[NUM_TWIG_PIECES] = {
"unknown",
"Twig"
};

static const char * TwigTypeImages[NUM_TWIG_PIECES] = {
"unknown",
":/icons/unknown.png"
};

static const char * TwigTypeIcons[NUM_TWIG_PIECES] = {
":/icons/unknown.png",
":/icons/twig.png"
};

static const char * TwigTypeDescs[NUM_TWIG_PIECES] = {
"unknown",
"synthesized from one stem and many leaves \n # deviations derived from input stem\n height unknown unit"
};

static inline int ToTwigType(int x) {
	return x - 192;
}

static const int TwigInPortRange[NUM_TWIG_PIECES][2] = {
{0,0},
{0,1},
};

static const char * TwigInPortRangeNames[2] = {
"inStem",
""
};

static const int TwigOutPortRange[NUM_TWIG_PIECES][2] = {
{0,0},
{0,1},
};

static const char * TwigOutPortRangeNames[2] = {
"outStem",
""
};

static const int TwigGeomDeviations[NUM_TWIG_PIECES] = {
0,
1,
};

}
#endif
