/*
 *  calcHarmonicCoordCmd.h
 *  calcHarmonicCoord
 *
 *  Created by jian zhang on 12/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <maya/MPxCommand.h>
#include <maya/MStatus.h>
#include <maya/MArgList.h>
#include <maya/MObject.h>
#include <maya/MDagPath.h>
#include <maya/MStringArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MSelectionList.h>

class HarmonicCoordCmd : public MPxCommand
{
public:
                HarmonicCoordCmd();
    virtual     ~HarmonicCoordCmd();

	MStatus		parseArgs( const MArgList& args );
    MStatus     doIt ( const MArgList& args );
    MStatus     redoIt ();
    MStatus     undoIt ();
    bool        isUndoable() const;

    static      void* creator();

private:
	int doubleArraySize(MPlug & plug, MDoubleArray & data) const;
	MString m_anchorArribName;
	MString m_valueArribName;
	MString m_meshName;
	bool m_showHelp;
};
