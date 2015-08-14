#include "HesperisCmd.h"
#include <maya/MGlobal.h>
#include <maya/MItSelectionList.h>
#include <maya/MItDag.h>
#include <maya/MArgDatabase.h>
#include <maya/MDagModifier.h>
#include "HesperisIO.h"
#include "HesperisFile.h"
#include <ASearchHelper.h>
#include <BaseTransform.h>
#include <H5FieldIn.h>
#include <HAttributeEntry.h>

void *HesperisCmd::creator()
{ return new HesperisCmd; }

MSyntax HesperisCmd::newSyntax() 
{
	MSyntax syntax;

	syntax.addFlag("-w", "-write", MSyntax::kString);
	syntax.addFlag("-gm", "-growMesh", MSyntax::kString);
    syntax.addFlag("-fd", "-fieldDeform", MSyntax::kString);
	syntax.addFlag("-h", "-help", MSyntax::kNoArg);
	syntax.enableQuery(false);
	syntax.enableEdit(false);

	return syntax;
}

MStatus HesperisCmd::parseArgs ( const MArgList& args )
{
	m_ioMode = IOUnknown;
	m_fileName = "";
	m_growMeshName = "";
	
	MArgDatabase argData(syntax(), args);
	MStatus stat;
	if(argData.isFlagSet("-w")) 
	{
		argData.getFlagArgument("-w", 0, m_fileName);
		if(!stat) {
			MGlobal::displayInfo(" cannot parse -w flag");
			return MS::kFailure;
		}
		
		m_ioMode = IOWrite;
		MGlobal::displayInfo(MString(" hesperis will write to file ") + m_fileName);
	}
    
    if(argData.isFlagSet("-fd")) {
        argData.getFlagArgument("-fd", 0, m_fileName);
		if(!stat) {
			MGlobal::displayInfo(" cannot parse -fd flag");
			return MS::kFailure;
		}
        m_ioMode = IOFieldDeform;
        MGlobal::displayInfo(MString(" hesperis will add field deformer ") + m_fileName);
    }
	
	if(argData.isFlagSet("-gm")) 
	{
		argData.getFlagArgument("-gm", 0, m_growMeshName);
		if(!stat) {
			MGlobal::displayInfo(" cannot parse -gm flag");
			return MS::kFailure;
		}
		
		if(m_ioMode == IOWrite) MGlobal::displayInfo(MString(" hesperis will write grow mesh ") + m_growMeshName);
		if(m_ioMode == IOFieldDeform) MGlobal::displayInfo(MString(" hesperis will attach to grow mesh ") + m_growMeshName);
	}
	
	if(argData.isFlagSet("-h")) 
	{
		m_ioMode = IOHelp;
	}
	
	if(m_ioMode == IOUnknown) {
		MGlobal::displayInfo(" no valid arguments are set, use -h for help");
		return MS::kFailure;
	}
    
    if(m_ioMode == IOWrite && m_growMeshName == "") {
        MGlobal::displayInfo(" no -gm is set for export, use -h for help");
		return MS::kFailure;
    }
	
	return MS::kSuccess;
}

MStatus HesperisCmd::doIt(const MArgList &args)
{
	MStatus status = parseArgs( args );
	
	if( status != MS::kSuccess ) return status;
	
	if(m_ioMode == IOHelp) return printHelp();
	
	MSelectionList selList;
    MGlobal::getActiveSelectionList(selList);
    
	if(selList.length() < 1) {
		MGlobal::displayInfo(" Empty selction!");
		return MS::kSuccess;
	}
    
    if(m_ioMode == IOWrite) return writeSelected(selList);
    if(m_ioMode == IOFieldDeform) return deformSelected();
	
	return MS::kSuccess;
}

MStatus HesperisCmd::writeSelected(const MSelectionList & selList)
{
	MItSelectionList iter( selList );
	
	std::map<std::string, MDagPath > curves;
	MDagPathArray tms;
	
	for(; !iter.isDone(); iter.next()) {								
		MDagPath apath;		
		iter.getDagPath( apath );
		tms.append(apath);
		ASearchHelper::AllTypedPaths(curves, apath, MFn::kNurbsCurve);
	}
	
	if(curves.size() < 1) {
		MGlobal::displayInfo(" zero curve selction!");
		return MS::kSuccess;
	}
	
	HesperisFile hesf;
	bool fstat = hesf.create(m_fileName.asChar());
	if(!fstat) {
		MGlobal::displayWarning(MString(" cannot create file ")+ m_fileName);
		return MS::kSuccess;
	}
	
	HesperisIO::WriteTransforms(tms, &hesf);
	HesperisIO::WriteCurves(curves, &hesf);
	
	writeMesh(&hesf);
	
	MGlobal::displayInfo(" done.");
	
	return MS::kSuccess;
}

void HesperisCmd::writeMesh(HesperisFile * file)
{
	if(m_growMeshName.length() < 3) return;
	ASearchHelper searcher;
	MDagPath meshGrp;
	if(!searcher.dagByFullName(m_growMeshName.asChar(), meshGrp)) return;
	std::map<std::string, MDagPath > meshes;
	ASearchHelper::AllTypedPaths(meshes, meshGrp, MFn::kMesh);
	if(meshes.size() < 1)
		MGlobal::displayInfo(MString(" no mesh found by name ")+m_growMeshName);

	HesperisIO::WriteMeshes(meshes, file);
}

MStatus HesperisCmd::printHelp()
{
	MGlobal::displayInfo(MString(" howto use hesperis cmd:")
         + MString("\n export mode")
		+MString("\n select group of curves to export")
		+MString("\n hesperis -w filename -gm fullPathToTransformOfMesh")
        +MString("\n -gm or -growMesh is full path name to the transform of grow mesh")
            + MString("\n deform mode")
            + MString("\n select group of geometries to deform")
            + MString("\n hesperis -fd filename -gm fullPathToTransformOfMesh")
            + MString("\n -fd or -fieldDeform is filename of .fld file") 
			+ MString("\n -gm or -growMesh is full path name to the transform of grow mesh that the deformed stuff will be attached to"));
	return MS::kSuccess;
}

void HesperisCmd::testTransform()
{
	ASearchHelper searcher;
	MDagPath meshGrp;
	if(!searcher.dagByFullName("|pCube1", meshGrp)) return;
	
	BaseTransform data;
	HesperisIO::GetTransform(&data, meshGrp);
}

MStatus HesperisCmd::deformSelected()
{
    H5FieldIn t;
    if(!t.open(m_fileName.asChar())) {
        MGlobal::displayInfo(MString("cannot open file")
                             + m_fileName);
        return MS::kFailure;
    }
    
    MGlobal::displayInfo(MString("hesperis add field deformer from file")
                         +m_fileName);
						 
	int minFrame = t.FirstFrame;
    int maxFrame = t.LastFrame;
    
    HFlt3AttributeEntry transgrp(".translate");
    Vector3F fieldT;
    transgrp.load(&fieldT);
    AHelper::Info<Vector3F>("field translate is ", fieldT);
	t.close();
	
    MGlobal::displayInfo(MString("hesperis field deform frame range (")
                         +minFrame
                         +MString(",")
                         +maxFrame
                         +MString(")"));
	
    MStringArray deformerName;
    MGlobal::executeCommand("deformer -type hesperisDeformer", deformerName);
    MGlobal::displayInfo(MString("node ")+deformerName[0]);
    MGlobal::executeCommand(MString("setAttr -type \"string\" ")
                            + deformerName[0]
                            + MString(".cachePath \"")
                            + m_fileName
                            + "\"");
							
	MObject odef;
	if( !ASearchHelper::FirstDepNodeByName(odef, deformerName[0], MFn::kPluginDeformerNode) ) {
		MGlobal::displayInfo(MString("cannot find node by name ")
							+ deformerName[0]);
		return MS::kFailure;
	}
	
	MFnDependencyNode fdef(odef);
	fdef.findPlug("minFrame").setValue(minFrame);
	fdef.findPlug("maxFrame").setValue(maxFrame);
	AHelper::SimpleAnimation(fdef.findPlug("currentTime"), minFrame, maxFrame);
	
	if(m_growMeshName == "") {
		MGlobal::displayInfo(" no grow mesh is set");
		return MS::kSuccess;
	}
	return attachSelected(fieldT);
}
	
MStatus HesperisCmd::attachSelected(const Vector3F & offsetV)
{
	MGlobal::displayInfo(MString(" attach to grow mesh ") + m_growMeshName);
	MSelectionList selList;
    MGlobal::getActiveSelectionList(selList);
    
	MItSelectionList iter( selList );
	
	MDagPath apath;		
	iter.getDagPath( apath );
		
	MObject otrans = apath.node();
	if(!otrans.hasFn(MFn::kTransform)) {
		MGlobal::displayWarning("must select a transform/group to attach to grow mesh");
		return MS::kFailure;
	}
	
	ASearchHelper searcher;
	MDagPath meshGrp;
	if(!searcher.dagByFullName(m_growMeshName.asChar(), meshGrp)) {
		MGlobal::displayWarning(MString("cannot find grow mesh by name ")+m_growMeshName);
		return MS::kFailure;
	}
	MObject ogrow = meshGrp.node();
	if(!ogrow.hasFn(MFn::kTransform)) {
		MGlobal::displayWarning("-gm must be a transform/group");
		return MS::kFailure;
	}
	
	MStatus stat;
    MDGModifier modif;
	MDagModifier dmodif;
	MObject hestranslate = modif.createNode("hesperisTranslateNode", &stat);
	modif.doIt();
    
    if(hestranslate.isNull()) {
		MGlobal::displayWarning("cannot create hes translate node");
		return MS::kFailure;
	}
	
	MFnDependencyNode fhest(hestranslate);
	MFnDependencyNode fgrow(ogrow);
	
	modif.connect(fgrow.findPlug("boundingBoxMinX", true), fhest.findPlug("bBoxMinX", true));
	modif.connect(fgrow.findPlug("boundingBoxMinY", true), fhest.findPlug("bBoxMinY", true));
	modif.connect(fgrow.findPlug("boundingBoxMinZ", true), fhest.findPlug("bBoxMinZ", true));
	modif.connect(fgrow.findPlug("boundingBoxMaxX", true), fhest.findPlug("bBoxMaxX", true));
	modif.connect(fgrow.findPlug("boundingBoxMaxY", true), fhest.findPlug("bBoxMaxY", true));
	modif.connect(fgrow.findPlug("boundingBoxMaxZ", true), fhest.findPlug("bBoxMaxZ", true));
	
	MPlug psrcwpmat = fgrow.findPlug("parentMatrix", true, &stat);
	if(!stat) MGlobal::displayInfo("cannot find plug worldParentMatrix");
	modif.connect(psrcwpmat, fhest.findPlug("inParentMatrix", true));
	modif.doIt();
	
    MFnDependencyNode ftrans(otrans);

	dmodif.connect(fhest.findPlug("outTranslateX", true), ftrans.findPlug("translateX", true));
	dmodif.connect(fhest.findPlug("outTranslateY", true), ftrans.findPlug("translateY", true));
	dmodif.connect(fhest.findPlug("outTranslateZ", true), ftrans.findPlug("translateZ", true));
	stat = dmodif.doIt();
	if(!stat) MGlobal::displayInfo(MString("cannot make some connections to ")+ftrans.name());
    
    fhest.findPlug("offsetX").setValue((double)-offsetV.x);
    fhest.findPlug("offsetY").setValue((double)-offsetV.y);
    fhest.findPlug("offsetZ").setValue((double)-offsetV.z);
    return MS::kSuccess;
}
//:~