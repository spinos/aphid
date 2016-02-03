/*
 *  BoxPaintContextCmd.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <maya/MPxContextCommand.h>
class proxyPaintContext;

class proxyPaintContextCmd : public MPxContextCommand
{
public:	
						proxyPaintContextCmd();
	virtual MStatus		doEditFlags();
	virtual MStatus doQueryFlags();
	virtual MPxContext*	makeObj();
	static	void*		creator();
	virtual MStatus		appendSyntax();
	
protected:
    proxyPaintContext*		fContext;
};