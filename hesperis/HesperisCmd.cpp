#include "HesperisCmd.h"
#include <maya/MGlobal.h>
#include <maya/MItSelectionList.h>
#include <maya/MItDag.h>
#include <maya/MArgDatabase.h>
#include "HesperisIO.h"
#include "HesperisFile.h"
#include <ASearchHelper.h>
#include <BaseTransform.h>
#include <H5FieldIn.h>

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
	
	if(argData.isFlagSet("-gm")) 
	{
		argData.getFlagArgument("-gm", 0, m_growMeshName);
		if(!stat) {
			MGlobal::displayInfo(" cannot parse -gm flag");
			return MS::kFailure;
		}
		
		MGlobal::displayInfo(MString(" hesperis will write grow mesh ") + m_growMeshName);
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
            + MString("\n hesperis -fd filename")
            + MString("\n -fd or -fieldDeform is filename of .fld file") );
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
    MStringArray deformerName;
    MGlobal::executeCommand("deformer -type hesperisDeformer", deformerName);
    MGlobal::displayInfo(MString("node ")+deformerName[0]);
    MGlobal::executeCommand(MString("setAttr -type \"string\" ")
                            + deformerName[0]
                            + MString(".cachePath \"")
                            + m_fileName
                            + "\"");
    
    int minFrame = t.FirstFrame;
    int maxFrame = t.LastFrame;
    MGlobal::displayInfo(MString("hesperis set field deformer frame range (")
                         +minFrame
                         +MString(",")
                         +maxFrame
                         +MString(")"));
    MGlobal::executeCommand(MString("setAttr ")
                            + deformerName[0]
                            + MString(".minFrame \"")
                            + minFrame
                            + "\"");
    MGlobal::executeCommand(MString("setAttr ")
                            + deformerName[0]
                            + MString(".maxFrame \"")
                            + maxFrame
                            + "\"");
    
    MGlobal::executeCommand(MString("setKeyframe -v ")
                            +minFrame
                            +MString(" -f ")
                            +minFrame
                            +MString(" -itt \"linear\"  -ott \"linear\" ")
                            +deformerName[0]
                            +MString(".currentTime"));
    MGlobal::executeCommand(MString("setKeyframe -v ")
                            +maxFrame
                            +MString(" -f ")
                            +maxFrame
                            +MString(" -itt \"linear\"  -ott \"linear\" ")
                            +deformerName[0]
                            +MString(".currentTime"));
    return MS::kSuccess;
}
//:~