/*
 *  branch.h
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_BRANCH_H
#define GAR_BRANCH_H

namespace gar {
    
#define NUM_BRANCH_PIECES 2

static const char * BranchTypeNames[NUM_BRANCH_PIECES] = {
"unknown",
"Simple Branch"
};

static const char * BranchTypeImages[NUM_BRANCH_PIECES] = {
"unknown",
":/icons/unknown.png"
};

static const char * BranchTypeIcons[NUM_BRANCH_PIECES] = {
":/icons/unknown.png",
":/icons/connect_branch.png"
};

static const char * BranchTypeDescs[NUM_BRANCH_PIECES] = {
"unknown",
"synthesized by connecting input stem \n # deviations unknown\n height unknown unit"
};

static inline int ToBranchType(int x) {
	return x - 224;
}

static const int BranchInPortRange[NUM_BRANCH_PIECES][2] = {
{0,0},
{0,1},
};

static const char * BranchInPortRangeNames[2] = {
"inStem",
"inLeaf",
};

static const int BranchOutPortRange[NUM_BRANCH_PIECES][2] = {
{0,0},
{0,1},
};

static const char * BranchOutPortRangeNames[2] = {
"outStem",
""
};

static const int BranchGeomDeviations[NUM_BRANCH_PIECES] = {
0,
1,
};

}
#endif
