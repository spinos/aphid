/*
 *  wbg_common.h
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef WBG_COMMON_H
#define WBG_COMMON_H

namespace wbg {

enum ShowComponent {
	scDefault = 0,
	scHeightField = 1
};

enum ToolAction {
	actNone = 0,
	actViewTop = 1,
	actViewPersp = 2
};

enum HeightFieldContext {
	hfcNone = 0,
	hfcMove = 1,
	hfcRotate = 2,
	hfcResize = 3
};

}
#endif