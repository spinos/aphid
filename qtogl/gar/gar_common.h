/*
 *  gar_common.h
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_COMMON_H
#define GAR_COMMON_H

namespace gar {

enum ToolAction {
	actViewGraph = 0,
	actViewPlant = 1
};

enum GrassTyp {
	gsNone = 0,
	gsClover = 1
};

static const char * PieceMimeStr = "image/x-garden-piece";

}
#endif