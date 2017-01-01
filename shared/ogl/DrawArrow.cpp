/*
 *  DrawArrow.cpp
 *  
 *
 *  Created by jian zhang on 1/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawArrow.h"
#include <math/Matrix44F.h>
#include <gl_heads.h>

namespace aphid {

static const int sXArrowNumTriangleFVVertices = 66;
static const float sXArrowTriangleFVVertices[] = {0.020067f, -0.021930f, 0.021930f,
0.775657f, -0.021930f, 0.021930f,
0.020067f, 0.021930f, 0.021930f,
0.020067f, 0.021930f, 0.021930f,
0.775657f, -0.021930f, 0.021930f,
0.775657f, 0.021930f, 0.021930f,
0.020067f, 0.021930f, 0.021930f,
0.775657f, 0.021930f, 0.021930f,
0.020067f, 0.021930f, -0.021930f,
0.020067f, 0.021930f, -0.021930f,
0.775657f, 0.021930f, 0.021930f,
0.775657f, 0.021930f, -0.021930f,
0.020067f, 0.021930f, -0.021930f,
0.775657f, 0.021930f, -0.021930f,
0.020067f, -0.021930f, -0.021930f,
0.020067f, -0.021930f, -0.021930f,
0.775657f, 0.021930f, -0.021930f,
0.775657f, -0.021930f, -0.021930f,
0.020067f, -0.021930f, -0.021930f,
0.775657f, -0.021930f, -0.021930f,
0.020067f, -0.021930f, 0.021930f,
0.020067f, -0.021930f, 0.021930f,
0.775657f, -0.021930f, -0.021930f,
0.775657f, -0.021930f, 0.021930f,
0.020067f, -0.021930f, -0.021930f,
0.020067f, -0.021930f, 0.021930f,
0.020067f, 0.021930f, -0.021930f,
0.020067f, 0.021930f, -0.021930f,
0.020067f, -0.021930f, 0.021930f,
0.020067f, 0.021930f, 0.021930f,
0.775657f, -0.021930f, 0.021930f,
0.775657f, -0.021930f, -0.021930f,
0.775864f, -0.047242f, 0.047242f,
0.775864f, -0.047242f, 0.047242f,
0.775657f, -0.021930f, -0.021930f,
0.775864f, -0.047242f, -0.047242f,
0.775657f, -0.021930f, -0.021930f,
0.775657f, 0.021930f, -0.021930f,
0.775864f, -0.047242f, -0.047242f,
0.775864f, -0.047242f, -0.047242f,
0.775657f, 0.021930f, -0.021930f,
0.775864f, 0.047242f, -0.047242f,
0.775657f, 0.021930f, -0.021930f,
0.775657f, 0.021930f, 0.021930f,
0.775864f, 0.047242f, -0.047242f,
0.775864f, 0.047242f, -0.047242f,
0.775657f, 0.021930f, 0.021930f,
0.775864f, 0.047242f, 0.047242f,
0.775657f, 0.021930f, 0.021930f,
0.775657f, -0.021930f, 0.021930f,
0.775864f, 0.047242f, 0.047242f,
0.775864f, 0.047242f, 0.047242f,
0.775657f, -0.021930f, 0.021930f,
0.775864f, -0.047242f, 0.047242f,
0.775864f, -0.047242f, -0.047242f,
0.775864f, 0.047242f, -0.047242f,
0.999405f, 0.000000f, 0.000000f,
0.775864f, -0.047242f, 0.047242f,
0.775864f, -0.047242f, -0.047242f,
0.999405f, 0.000000f, 0.000000f,
0.775864f, 0.047242f, 0.047242f,
0.775864f, -0.047242f, 0.047242f,
0.999405f, 0.000000f, 0.000000f,
0.775864f, 0.047242f, -0.047242f,
0.775864f, 0.047242f, 0.047242f,
0.999405f, 0.000000f, 0.000000f
};
static const float sXArrowTriangleFVNormals[] = {0.000000f, 0.000000f, 1.000000f,
0.000000f, 0.000000f, 1.000000f,
0.000000f, 0.000000f, 1.000000f,
-0.000000f, 0.000000f, 1.000000f,
-0.000000f, 0.000000f, 1.000000f,
-0.000000f, 0.000000f, 1.000000f,
-0.000000f, 1.000000f, 0.000000f,
-0.000000f, 1.000000f, 0.000000f,
-0.000000f, 1.000000f, 0.000000f,
0.000000f, 1.000000f, 0.000000f,
0.000000f, 1.000000f, 0.000000f,
0.000000f, 1.000000f, 0.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-0.999967f, -0.008162f, -0.000000f,
-0.999967f, -0.008162f, -0.000000f,
-0.999967f, -0.008162f, -0.000000f,
-0.999967f, -0.008162f, -0.000000f,
-0.999967f, -0.008162f, -0.000000f,
-0.999967f, -0.008162f, -0.000000f,
-0.999967f, 0.000000f, -0.008162f,
-0.999967f, 0.000000f, -0.008162f,
-0.999967f, 0.000000f, -0.008162f,
-0.999967f, 0.000000f, -0.008162f,
-0.999967f, 0.000000f, -0.008162f,
-0.999967f, 0.000000f, -0.008162f,
-0.999967f, 0.008162f, 0.000000f,
-0.999967f, 0.008162f, 0.000000f,
-0.999967f, 0.008162f, 0.000000f,
-0.999967f, 0.008162f, 0.000000f,
-0.999967f, 0.008162f, 0.000000f,
-0.999967f, 0.008162f, 0.000000f,
-0.999967f, 0.000000f, 0.008162f,
-0.999967f, 0.000000f, 0.008162f,
-0.999967f, 0.000000f, 0.008162f,
-0.999967f, 0.000000f, 0.008162f,
-0.999967f, 0.000000f, 0.008162f,
-0.999967f, 0.000000f, 0.008162f,
0.206766f, 0.000000f, -0.978390f,
0.206766f, 0.000000f, -0.978390f,
0.206766f, 0.000000f, -0.978390f,
0.206766f, -0.978390f, 0.000000f,
0.206766f, -0.978390f, 0.000000f,
0.206766f, -0.978390f, 0.000000f,
0.206766f, 0.000000f, 0.978390f,
0.206766f, 0.000000f, 0.978390f,
0.206766f, 0.000000f, 0.978390f,
0.206766f, 0.978390f, -0.000000f,
0.206766f, 0.978390f, -0.000000f,
0.206766f, 0.978390f, -0.000000f
};

static const int sYArrowNumTriangleFVVertices = 66;
static const float sYArrowTriangleFVVertices[] = {0.021930f, 0.020067f, 0.021930f,
0.021930f, 0.775657f, 0.021930f,
-0.021930f, 0.020067f, 0.021930f,
-0.021930f, 0.020067f, 0.021930f,
0.021930f, 0.775657f, 0.021930f,
-0.021930f, 0.775657f, 0.021930f,
-0.021930f, 0.020067f, 0.021930f,
-0.021930f, 0.775657f, 0.021930f,
-0.021930f, 0.020067f, -0.021930f,
-0.021930f, 0.020067f, -0.021930f,
-0.021930f, 0.775657f, 0.021930f,
-0.021930f, 0.775657f, -0.021930f,
-0.021930f, 0.020067f, -0.021930f,
-0.021930f, 0.775657f, -0.021930f,
0.021930f, 0.020067f, -0.021930f,
0.021930f, 0.020067f, -0.021930f,
-0.021930f, 0.775657f, -0.021930f,
0.021930f, 0.775657f, -0.021930f,
0.021930f, 0.020067f, -0.021930f,
0.021930f, 0.775657f, -0.021930f,
0.021930f, 0.020067f, 0.021930f,
0.021930f, 0.020067f, 0.021930f,
0.021930f, 0.775657f, -0.021930f,
0.021930f, 0.775657f, 0.021930f,
0.021930f, 0.020067f, -0.021930f,
0.021930f, 0.020067f, 0.021930f,
-0.021930f, 0.020067f, -0.021930f,
-0.021930f, 0.020067f, -0.021930f,
0.021930f, 0.020067f, 0.021930f,
-0.021930f, 0.020067f, 0.021930f,
0.021930f, 0.775657f, 0.021930f,
0.021930f, 0.775657f, -0.021930f,
0.047242f, 0.775864f, 0.047242f,
0.047242f, 0.775864f, 0.047242f,
0.021930f, 0.775657f, -0.021930f,
0.047242f, 0.775864f, -0.047242f,
0.021930f, 0.775657f, -0.021930f,
-0.021930f, 0.775657f, -0.021930f,
0.047242f, 0.775864f, -0.047242f,
0.047242f, 0.775864f, -0.047242f,
-0.021930f, 0.775657f, -0.021930f,
-0.047242f, 0.775864f, -0.047242f,
-0.021930f, 0.775657f, -0.021930f,
-0.021930f, 0.775657f, 0.021930f,
-0.047242f, 0.775864f, -0.047242f,
-0.047242f, 0.775864f, -0.047242f,
-0.021930f, 0.775657f, 0.021930f,
-0.047242f, 0.775864f, 0.047242f,
-0.021930f, 0.775657f, 0.021930f,
0.021930f, 0.775657f, 0.021930f,
-0.047242f, 0.775864f, 0.047242f,
-0.047242f, 0.775864f, 0.047242f,
0.021930f, 0.775657f, 0.021930f,
0.047242f, 0.775864f, 0.047242f,
0.047242f, 0.775864f, -0.047242f,
-0.047242f, 0.775864f, -0.047242f,
0.000000f, 0.999405f, 0.000000f,
0.047242f, 0.775864f, 0.047242f,
0.047242f, 0.775864f, -0.047242f,
0.000000f, 0.999405f, 0.000000f,
-0.047242f, 0.775864f, 0.047242f,
0.047242f, 0.775864f, 0.047242f,
0.000000f, 0.999405f, 0.000000f,
-0.047242f, 0.775864f, -0.047242f,
-0.047242f, 0.775864f, 0.047242f,
0.000000f, 0.999405f, 0.000000f
};
static const float sYArrowTriangleFVNormals[] = {0.000000f, 0.000000f, 1.000000f,
0.000000f, 0.000000f, 1.000000f,
0.000000f, 0.000000f, 1.000000f,
0.000000f, 0.000000f, 1.000000f,
0.000000f, 0.000000f, 1.000000f,
0.000000f, 0.000000f, 1.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
0.000000f, 0.000000f, -1.000000f,
1.000000f, 0.000000f, -0.000000f,
1.000000f, 0.000000f, -0.000000f,
1.000000f, 0.000000f, -0.000000f,
1.000000f, 0.000000f, 0.000000f,
1.000000f, 0.000000f, 0.000000f,
1.000000f, 0.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.008161f, -0.999967f, 0.000000f,
0.008161f, -0.999967f, 0.000000f,
0.008161f, -0.999967f, 0.000000f,
0.008161f, -0.999967f, 0.000000f,
0.008161f, -0.999967f, 0.000000f,
0.008161f, -0.999967f, 0.000000f,
-0.000000f, -0.999967f, -0.008162f,
-0.000000f, -0.999967f, -0.008162f,
-0.000000f, -0.999967f, -0.008162f,
-0.000000f, -0.999967f, -0.008162f,
-0.000000f, -0.999967f, -0.008162f,
-0.000000f, -0.999967f, -0.008162f,
-0.008161f, -0.999967f, 0.000000f,
-0.008161f, -0.999967f, 0.000000f,
-0.008161f, -0.999967f, 0.000000f,
-0.008161f, -0.999967f, 0.000000f,
-0.008161f, -0.999967f, 0.000000f,
-0.008161f, -0.999967f, 0.000000f,
0.000000f, -0.999967f, 0.008162f,
0.000000f, -0.999967f, 0.008162f,
0.000000f, -0.999967f, 0.008162f,
0.000000f, -0.999967f, 0.008162f,
0.000000f, -0.999967f, 0.008162f,
0.000000f, -0.999967f, 0.008162f,
0.000000f, 0.206766f, -0.978390f,
0.000000f, 0.206766f, -0.978390f,
0.000000f, 0.206766f, -0.978390f,
0.978390f, 0.206766f, 0.000000f,
0.978390f, 0.206766f, 0.000000f,
0.978390f, 0.206766f, 0.000000f,
-0.000000f, 0.206766f, 0.978390f,
-0.000000f, 0.206766f, 0.978390f,
-0.000000f, 0.206766f, 0.978390f,
-0.978390f, 0.206766f, 0.000000f,
-0.978390f, 0.206766f, 0.000000f,
-0.978390f, 0.206766f, 0.000000f
};

static const int sZArrowNumTriangleFVVertices = 66;
static const float sZArrowTriangleFVVertices[] = {0.021930f, -0.021930f, 0.020067f,
0.021930f, -0.021930f, 0.775657f,
-0.021930f, -0.021930f, 0.020067f,
-0.021930f, -0.021930f, 0.020067f,
0.021930f, -0.021930f, 0.775657f,
-0.021930f, -0.021930f, 0.775657f,
-0.021930f, -0.021930f, 0.020067f,
-0.021930f, -0.021930f, 0.775657f,
-0.021930f, 0.021930f, 0.020067f,
-0.021930f, 0.021930f, 0.020067f,
-0.021930f, -0.021930f, 0.775657f,
-0.021930f, 0.021930f, 0.775657f,
-0.021930f, 0.021930f, 0.020067f,
-0.021930f, 0.021930f, 0.775657f,
0.021930f, 0.021930f, 0.020067f,
0.021930f, 0.021930f, 0.020067f,
-0.021930f, 0.021930f, 0.775657f,
0.021930f, 0.021930f, 0.775657f,
0.021930f, 0.021930f, 0.020067f,
0.021930f, 0.021930f, 0.775657f,
0.021930f, -0.021930f, 0.020067f,
0.021930f, -0.021930f, 0.020067f,
0.021930f, 0.021930f, 0.775657f,
0.021930f, -0.021930f, 0.775657f,
0.021930f, 0.021930f, 0.020067f,
0.021930f, -0.021930f, 0.020067f,
-0.021930f, 0.021930f, 0.020067f,
-0.021930f, 0.021930f, 0.020067f,
0.021930f, -0.021930f, 0.020067f,
-0.021930f, -0.021930f, 0.020067f,
0.021930f, -0.021930f, 0.775657f,
0.021930f, 0.021930f, 0.775657f,
0.047242f, -0.047242f, 0.775864f,
0.047242f, -0.047242f, 0.775864f,
0.021930f, 0.021930f, 0.775657f,
0.047242f, 0.047242f, 0.775864f,
0.021930f, 0.021930f, 0.775657f,
-0.021930f, 0.021930f, 0.775657f,
0.047242f, 0.047242f, 0.775864f,
0.047242f, 0.047242f, 0.775864f,
-0.021930f, 0.021930f, 0.775657f,
-0.047242f, 0.047242f, 0.775864f,
-0.021930f, 0.021930f, 0.775657f,
-0.021930f, -0.021930f, 0.775657f,
-0.047242f, 0.047242f, 0.775864f,
-0.047242f, 0.047242f, 0.775864f,
-0.021930f, -0.021930f, 0.775657f,
-0.047242f, -0.047242f, 0.775864f,
-0.021930f, -0.021930f, 0.775657f,
0.021930f, -0.021930f, 0.775657f,
-0.047242f, -0.047242f, 0.775864f,
-0.047242f, -0.047242f, 0.775864f,
0.021930f, -0.021930f, 0.775657f,
0.047242f, -0.047242f, 0.775864f,
0.047242f, 0.047242f, 0.775864f,
-0.047242f, 0.047242f, 0.775864f,
0.000000f, -0.000000f, 0.999405f,
0.047242f, -0.047242f, 0.775864f,
0.047242f, 0.047242f, 0.775864f,
0.000000f, -0.000000f, 0.999405f,
-0.047242f, -0.047242f, 0.775864f,
0.047242f, -0.047242f, 0.775864f,
0.000000f, -0.000000f, 0.999405f,
-0.047242f, 0.047242f, 0.775864f,
-0.047242f, -0.047242f, 0.775864f,
0.000000f, -0.000000f, 0.999405f
};
static const float sZArrowTriangleFVNormals[] = {0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, 0.000000f,
0.000000f, -1.000000f, -0.000000f,
0.000000f, -1.000000f, -0.000000f,
0.000000f, -1.000000f, -0.000000f,
-1.000000f, -0.000000f, 0.000000f,
-1.000000f, -0.000000f, 0.000000f,
-1.000000f, -0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-1.000000f, 0.000000f, 0.000000f,
-0.000000f, 1.000000f, 0.000000f,
-0.000000f, 1.000000f, 0.000000f,
-0.000000f, 1.000000f, 0.000000f,
-0.000000f, 1.000000f, -0.000000f,
-0.000000f, 1.000000f, -0.000000f,
-0.000000f, 1.000000f, -0.000000f,
1.000000f, -0.000000f, 0.000000f,
1.000000f, -0.000000f, 0.000000f,
1.000000f, -0.000000f, 0.000000f,
1.000000f, 0.000000f, 0.000000f,
1.000000f, 0.000000f, 0.000000f,
1.000000f, 0.000000f, 0.000000f,
-0.000000f, 0.000000f, -1.000000f,
-0.000000f, 0.000000f, -1.000000f,
-0.000000f, 0.000000f, -1.000000f,
-0.000000f, 0.000000f, -1.000000f,
-0.000000f, 0.000000f, -1.000000f,
-0.000000f, 0.000000f, -1.000000f,
0.008161f, 0.000000f, -0.999967f,
0.008161f, 0.000000f, -0.999967f,
0.008161f, 0.000000f, -0.999967f,
0.008161f, 0.000000f, -0.999967f,
0.008161f, 0.000000f, -0.999967f,
0.008161f, 0.000000f, -0.999967f,
-0.000000f, 0.008162f, -0.999967f,
-0.000000f, 0.008162f, -0.999967f,
-0.000000f, 0.008162f, -0.999967f,
0.000000f, 0.008162f, -0.999967f,
0.000000f, 0.008162f, -0.999967f,
0.000000f, 0.008162f, -0.999967f,
-0.008161f, -0.000000f, -0.999967f,
-0.008161f, -0.000000f, -0.999967f,
-0.008161f, -0.000000f, -0.999967f,
-0.008161f, -0.000000f, -0.999967f,
-0.008161f, -0.000000f, -0.999967f,
-0.008161f, -0.000000f, -0.999967f,
0.000000f, -0.008162f, -0.999967f,
0.000000f, -0.008162f, -0.999967f,
0.000000f, -0.008162f, -0.999967f,
0.000000f, -0.008162f, -0.999967f,
0.000000f, -0.008162f, -0.999967f,
0.000000f, -0.008162f, -0.999967f,
0.000000f, 0.978390f, 0.206766f,
0.000000f, 0.978390f, 0.206766f,
0.000000f, 0.978390f, 0.206766f,
0.978390f, -0.000000f, 0.206766f,
0.978390f, -0.000000f, 0.206766f,
0.978390f, -0.000000f, 0.206766f,
0.000000f, -0.978390f, 0.206766f,
0.000000f, -0.978390f, 0.206766f,
0.000000f, -0.978390f, 0.206766f,
-0.978390f, 0.000000f, 0.206766f,
-0.978390f, 0.000000f, 0.206766f,
-0.978390f, 0.000000f, 0.206766f
};


DrawArrow::DrawArrow()
{}

DrawArrow::~DrawArrow()
{}

void DrawArrow::drawArrowAt(const Matrix44F * mat)
{
	glPushMatrix();
	
	float transbuf[16];
	mat->glMatrix(transbuf);
	glMultMatrixf((const GLfloat*)transbuf);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)sXArrowTriangleFVNormals);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sXArrowTriangleFVVertices);
	glDrawArrays(GL_TRIANGLES, 0, sXArrowNumTriangleFVVertices);
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPopMatrix();
	
}

void DrawArrow::drawCoordinateAt(const Matrix44F * mat)
{
	glPushMatrix();
	
	float transbuf[16];
	mat->glMatrix(transbuf);
	glMultMatrixf((const GLfloat*)transbuf);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	glColor3f(1,0,0);
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)sXArrowTriangleFVNormals);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sXArrowTriangleFVVertices);
	
	glDrawArrays(GL_TRIANGLES, 0, sXArrowNumTriangleFVVertices);
	
	glColor3f(0,1,0);
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)sYArrowTriangleFVNormals);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sYArrowTriangleFVVertices);
	
	glDrawArrays(GL_TRIANGLES, 0, sYArrowNumTriangleFVVertices);
	
	glColor3f(0,0,1);
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)sZArrowTriangleFVNormals);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sZArrowTriangleFVVertices);
	glDrawArrays(GL_TRIANGLES, 0, sZArrowNumTriangleFVVertices);
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPopMatrix();
}

}