#include "UtilityCmd.h"
#include <maya/MObject.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MItSelectionList.h>
#include <maya/MSelectionList.h>
#include <maya/MObjectArray.h>
#include <maya/MItDependencyNodes.h>
#include <maya/MDagModifier.h>
#include <Mdag.h>
#include <Marg.h>
#include <Mplg.h>
#include "ConditionNode.h"
#include "RigidBodyTransform.h"
#include <Array.h>

namespace caterpillar {

static const char * attachFlag			= "-at";
static const char * attachFlagLong		= "-attach";
static const char * lsGroupFlag			= "-lgn";
static const char * lsGroupFlagLong		= "-lsGroupName";
static const char * groupNameFlag		= "-gn";
static const char * groupNameFlagLong	= "-groupName";
static const char * modelNameFlag		= "-mdn";
static const char * modelNameFlagLong	= "-modelName";
static const int numRegisteredNames = 2;
static const MString registeredNames[] = {"caterpillarCondition", "caterpillarTrackedVehicle"};

UtilityCmd::UtilityCmd() 
{
	sdb::TreeNode::MaxNumKeysPerNode = 256;
	sdb::TreeNode::MinNumKeysPerNode = 128;
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
	status = marg::AddFlag(syntax, lsGroupFlag, lsGroupFlagLong, MSyntax::kNoArg);
	status = marg::AddFlag(syntax, groupNameFlag, groupNameFlagLong, MSyntax::kString);
	status = marg::AddFlag(syntax, modelNameFlag, modelNameFlagLong, MSyntax::kString);

	return syntax;
}

MStatus UtilityCmd::parseArgs( const MArgList& args )
{
    m_operation = tUnknown;
	m_groupName = "";

	MArgDatabase argData(syntax(), args);
	
	if(marg::HasFlag(argData, attachFlag)) {
		return checkAttachOpt(args);
	}
	else if(marg::HasFlag(argData, lsGroupFlag)) {
	    return checkListOpt(args);
	}
	
	MGlobal::displayWarning("caterpillar has no -attach  or -lsGroupName flag");
	return MS::kFailure;
}

MStatus UtilityCmd::checkAttachOpt(const MArgList& args)
{
    MString conditionName, modelName;
	MStatus     	status;
	MArgDatabase argData(syntax(), args);
	
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
	m_operation = tAttachModel;
	return MS::kSuccess;
}

MStatus UtilityCmd::checkListOpt(const MArgList& args)
{
    MString conditionName;
    if(!marg::GetLast<MString>(args, conditionName)) {
		MGlobal::displayWarning("caterpillar has no subject");
		return MS::kFailure;
	}
	
	if(!mdag::FindObjByName(conditionName, m_conditionNode)) {
		MGlobal::displayWarning(MString("caterpillar cannot find condition ") + conditionName);
		return MS::kFailure;
	}
	m_operation = tLsGroup;
    return MS::kSuccess;
}

MStatus UtilityCmd::doIt( const MArgList& args )
{
    MStatus stat = parseArgs(args);
	if (stat != MS::kSuccess) {
	    return stat;
	}
	return redoIt();
}

MStatus UtilityCmd::redoIt()
{
    if(m_operation == tLsGroup) return doLsGroup();
    else if(m_operation == tAttachModel) return doAttachModel();
    return MS::kFailure;
}

MStatus UtilityCmd::doLsGroup()
{
    if(m_conditionNode == MObject::kNullObj) return MS::kFailure;
    MFnDependencyNode fcondition(m_conditionNode);
    
    GroupId * g = getGroupId();
	if(!g) {
		MGlobal::displayWarning(MString("caterpillar cannot cast groupId from ")+fcondition.name());
		return MS::kFailure;
	}
	
	const std::deque<std::string > allNames = g->getGroupNames();
	MStringArray res;
	std::deque<std::string >::const_iterator it = allNames.begin();
	for(; it != allNames.end(); ++it) 
	    res.append(MString((*it).c_str()));
	setResult(res);
	MGlobal::displayInfo(MString("caterpillar list group names in ") + fcondition.name());
    return MS::kSuccess;
}

MStatus UtilityCmd::doAttachModel()
{
	if(m_conditionNode == MObject::kNullObj || m_modelNode == MObject::kNullObj) return MS::kFailure;
	MFnDependencyNode fcondition(m_conditionNode);
	MStatus status;
	MPlug pos = fcondition.findPlug("outSolver", &status);
	if(!status) {
		MGlobal::displayWarning(MString("caterpillar cannot find outSolver plug in ")+fcondition.name());
		return MS::kFailure;
	}
	
	GroupId * g = getGroupId();
	if(!g) {
		MGlobal::displayWarning(MString("caterpillar cannot cast groupId from ")+fcondition.name());
		return MS::kFailure;
	}
	
	if(!g->hasGroup(m_groupName.asChar())) {
		MGlobal::displayWarning(MString("caterpillar cannot find group \"") + m_groupName + ("\" in ")+fcondition.name());
		return MS::kSuccess;
	}
	
	std::deque<int> & indices = g->group(m_groupName.asChar());
	
	if(indices.size() < 1) {
		MGlobal::displayWarning(MString("no object in group \"") + m_groupName + ("\" in ")+fcondition.name());
		return MS::kSuccess;
	}
	
	MObjectArray conn = mplg::GetOutgoingConnectedNodes(pos);
	if(conn.length() < 1) {
		MGlobal::displayWarning(MString("caterpillar cannot find outSolver connection in ")+fcondition.name());
		return MS::kFailure;
	}
	
	MObject solverNode = conn[0];
	MFnDependencyNode fsolver(solverNode);
	if(fsolver.typeName() != "caterpillarSolver") {
		MGlobal::displayWarning(fsolver.name() + " is not caterpillar solver");
		return MS::kFailure;
	}
	
	MPlug porb = fsolver.findPlug("outRigidBodies", &status);
	if(!status) {
		MGlobal::displayWarning(MString("caterpillar cannot find outRigidBodies plug in ")+fsolver.name());
		return MS::kFailure;
	}
	
	sdb::Array<int, MObject> transArray;
	
	MObjectArray rbs = mplg::GetOutgoingConnectedNodes(porb);
	for(int i=0; i < rbs.length(); i++) {
	    const int objId = getObjectId(rbs[i]);
	    if(objId > -1) transArray.insert(objId, &rbs[i]);
	}
	
	MObject parentGrp = mdag::CreateGroup(m_groupName);
	
	std::deque<int>::const_iterator it = indices.begin();
	for(; it != indices.end(); ++it) {
	    MObject * found = transArray.find(*it);
		MObject rbt;
	    if(found) rbt = *found;
	    else rbt = createRigidBody(porb, *it, parentGrp);
		
		if(rbt == MObject::kNullObj) continue;
	    MObject amdl = mdag::DuplicateDagNode(m_modelNode);
		if(amdl == MObject::kNullObj) {
			MGlobal::displayWarning(MString("caterpillar cannot duplicate "));
			continue;
		}
		mdag::ParentTo(amdl, rbt);
	}
	
	MGlobal::displayInfo(MString("caterpillar attached model to ") + indices.size() + " rigid bodies.");
    return MS::kSuccess;
}

GroupId * UtilityCmd::getGroupId() const
{
	MFnDependencyNode fcondition(m_conditionNode);
	for(int i = 0; i < numRegisteredNames; i++) {
		if(fcondition.typeName() == registeredNames[i]) {
			ConditionNode * g = (ConditionNode *)fcondition.userNode();
			return g;
		}
	}
	return NULL;
}

const int UtilityCmd::getObjectId(const MObject & node) const
{
    MFnDependencyNode fnode(node);
    MStatus status;
    MPlug p = fnode.findPlug("objectId", &status);
    if(!status) return -1;
    return p.asInt();
}

MObject UtilityCmd::createRigidBody(const MPlug & src, const int & objectId, MObject & parent) const
{
	MDagModifier mod;
	MStatus status;
	MObject result = mod.createNode(caterpillar::RigidBodyTransformNode::id, parent, &status);
    if(!status) {
		MGlobal::displayWarning("cannot create rigid body transform");
		return MObject::kNullObj;
	}
	mod.doIt();
    MFnDependencyNode fnode(result);
    fnode.findPlug("objectId").setValue(objectId);
    mod.connect(src, fnode.findPlug("inSolver"));
    mod.doIt();
	return result;
}

}
