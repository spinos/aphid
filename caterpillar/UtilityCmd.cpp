#include "UtilityCmd.h"
#include <maya/MObject.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MItSelectionList.h>
#include <maya/MSelectionList.h>
#include <maya/MObjectArray.h>
#include <maya/MItDependencyNodes.h>
#include <Mdag.h>
#include <Marg.h>
#include "ConditionNode.h"

namespace caterpillar {

static const char * attachFlag			= "-at";
static const char * attachFlagLong		= "-attach";
static const char * groupNameFlag		= "-gn";
static const char * groupNameFlagLong	= "-groupName";
static const char * modelNameFlag		= "-mdn";
static const char * modelNameFlagLong	= "-modelName";

UtilityCmd::UtilityCmd()
{
}

UtilityCmd::~UtilityCmd() {}

void* UtilityCmd::creator()
{
    return new UtilityCmd;
}

bool UtilityCmd::isUndoable() const
{
    return false;
}

MStatus UtilityCmd::undoIt()
{
    return MS::kSuccess;
}

MSyntax UtilityCmd::newSyntax ()
{
	MStatus status;

	MSyntax syntax;
	syntax.setMinObjects(1);
	syntax.setMaxObjects(1);
	syntax.setObjectType(MSyntax::kStringObjects);
	status = marg::AddFlag(syntax, attachFlag, attachFlagLong, MSyntax::kNoArg);
	status = marg::AddFlag(syntax, groupNameFlag, groupNameFlagLong, MSyntax::kString);
	status = marg::AddFlag(syntax, modelNameFlag, modelNameFlagLong, MSyntax::kString);

	return syntax;
}

MStatus UtilityCmd::parseArgs( const MArgList& args )
{
	m_groupName = "";
	MString conditionName, modelName;
	MStatus     	status;
	MArgDatabase argData(syntax(), args);
	
	if(!marg::HasFlag(argData, attachFlag)) {
		MGlobal::displayWarning("caterpillar has no -attach flag");
		return MS::kFailure;
	}
	
	if(!marg::GetLast<MString>(args, conditionName)) {
		MGlobal::displayWarning("caterpillar has no subject");
		return MS::kFailure;
	}
	
	if(!mdag::FindObjByName(conditionName, m_conditionNode)) {
		MGlobal::displayWarning(MString("caterpillar cannot find condition ") + conditionName);
		return MS::kFailure;
	}
	
	if(!marg::GetArgValue<MString>(argData, groupNameFlag, m_groupName)) {
		MGlobal::displayWarning("caterpillar has no -groupName flag");
		return MS::kFailure;
	}
	
	if(!marg::GetArgValue<MString>(argData, modelNameFlag, modelName)) {
		MGlobal::displayWarning("caterpillar has no -modelName flag");
		return MS::kFailure;
	}
	
	if(!mdag::FindObjByName(modelName, m_modelNode)) {
		MGlobal::displayWarning(MString("caterpillar cannot find model ") + modelName);
		return MS::kFailure;
	}

	/*
	MString     	arg;
	const MString	groupNameFlag			("-gn");
	const MString	groupNameFlagLong		("-groupName");
	const MString	modelNameFlag			("-mdn");
	const MString	modelNameFlagLong		("-modelName");
// Parse the arguments.
	for ( unsigned int i = 0; i < args.length(); i++ ) {
		arg = args.asString( i, &stat );
		if (!stat)              
			continue;
		
		if ( arg == groupNameFlag || arg == groupNameFlagLong ) {
			if (i == args.length()-1) 
				continue;
			i++;
			args.get(i, m_groupName);
		}
		
		if ( arg == modelNameFlag || arg == modelNameFlagLong ) {
			if (i == args.length()-1) 
				continue;
			i++;
			args.get(i, m_modelName);
		}
		
		if (i == args.length()-1)
			args.get(i, m_conditionName);
	}
	
	if(m_groupName == "") {
		MGlobal::displayWarning("caterpillar command has no -gn flag");
		return MS::kFailure;
	}
	
	if(m_modelName == "") {
		MGlobal::displayWarning("caterpillar command has no -mdn flag");
		return MS::kFailure;
	}
	
	if(m_conditionName == "") {
		MGlobal::displayWarning("caterpillar command has no subject condition");
		return MS::kFailure;
	}
*/
	return MS::kSuccess;
}


MStatus UtilityCmd::doIt( const MArgList& args )
{
    MStatus stat = parseArgs(args);
	if (stat != MS::kSuccess) {
		return stat;
	}
	if(m_conditionNode == MObject::kNullObj || m_modelNode == MObject::kNullObj) return stat;
	return redoIt();
}


MStatus UtilityCmd::redoIt()
{
	MFnDependencyNode fcondition(m_conditionNode);
	ConditionNode * g = (ConditionNode *)fcondition.userNode();
	if(!g) {
		MGlobal::displayWarning(MString("caterpillar cannot cast groupId from ")+fcondition.name());
		return MS::kSuccess;
	}
	
	if(!g->hasGroup(m_groupName.asChar())) {
		MGlobal::displayWarning(MString("caterpillar cannot find group \"") + m_groupName + ("\" in ")+fcondition.name());
		return MS::kSuccess;
	}
	
	const std::deque<int> & indices = g->group(m_groupName.asChar());
	MGlobal::displayInfo(MString("count ") + indices.size());
	
    return MS::kSuccess;
}
}
