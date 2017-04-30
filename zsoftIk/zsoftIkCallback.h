/*
 *  zsoftIkCallback.h
 *  softIk
 *
 *  Created by jian zhang on 3/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <maya/MPxCommand.h>
#include <maya/MSceneMessage.h>
class addIK2BsolverCallbacks : public MPxCommand {
public:                                                                                                                 
		addIK2BsolverCallbacks() {};
        virtual MStatus doIt (const MArgList &);
        static void*    creator();
        
        // callback IDs for the solver callbacks
        static MCallbackId afterNewId;
        static MCallbackId afterOpenId;
};
