/*
 *  UtilityCmd.h
 *
 *  Created by jian zhang on 1/8/13
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <maya/MPxCommand.h>
#include <maya/MStatus.h>
#include <maya/MArgList.h>
#include <maya/MString.h>
#include <maya/MSyntax.h>
#include <maya/MObject.h>
#include <maya/MPlug.h>

namespace caterpillar {
class GroupId;
class UtilityCmd : public MPxCommand
{
public:
                UtilityCmd();
    virtual     ~UtilityCmd();

	MStatus		parseArgs( const MArgList& args );
    MStatus     doIt ( const MArgList& args );
    MStatus     redoIt ();
    MStatus     undoIt ();
    bool        isUndoable() const;

    static      void* creator();
	static MSyntax newSyntax ();

private:
	GroupId * getGroupId() const;
	const int getObjectId(const MObject & node) const;
	MObject createRigidBody(const MPlug & src, const int & objectId, MObject & parent) const;
	MString m_groupName;
	MObject m_conditionNode;
	MObject m_modelNode;
};
}
