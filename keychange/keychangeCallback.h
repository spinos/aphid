/*
 *  zsoftIkCallback.h
 *  softIk
 *
 *  Created by jian zhang on 3/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <maya/MPxCommand.h>
#include <maya/MAnimMessage.h>
class keychagneCallbacks : public MPxCommand {
public:                                                                                                                 
		keychagneCallbacks() {};
        virtual MStatus doIt (const MArgList &);
        static void*    creator();
        
        static MCallbackId keyEditedId;
};
