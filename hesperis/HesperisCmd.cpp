#include "HesperisCmd.h"
#include <maya/MGlobal.h>
#include <maya/MSelectionList.h>
#include <maya/MItSelectionList.h>
#include <maya/MItDag.h>
#include <maya/MArgDatabase.h>
#include "HesperisIO.h"
#include "HesperisFile.h"
#include <ASearchHelper.h>

void *HesperisCmd::creator()
{ return new HesperisCmd; }

MSyntax HesperisCmd::newSyntax() 
{
	MSyntax syntax;

	syntax.addFlag("-w", "-write", MSyntax::kString);
	syntax.addFlag("-gm", "-growMesh", MSyntax::kString);
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
	
	if(argData.isFlagSet("-h")) 
	{
		m_ioMode = IOHelp;
	}
	
	if(m_ioMode == IOUnknown) {
		MGlobal::displayInfo(" no valid arguments are set, use -h for help");
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
	
	MItSelectionList iter( selList );
	
	MDagPathArray curves;
	MDagPathArray tms;
	
	for(; !iter.isDone(); iter.next()) {								
		MDagPath apath;		
		iter.getDagPath( apath );
		tms.append(apath);
		ASearchHelper::AllTypedPaths(apath, curves, MFn::kNurbsCurve);
	}
	
	if(curves.length() < 1) {
		MGlobal::displayInfo(" Zero curve selction!");
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
	MDagPathArray meshes;
	ASearchHelper::AllTypedPaths(meshGrp, meshes, MFn::kMesh);
	if(meshes.length() < 1)
		MGlobal::displayInfo(MString(" no mesh found by name ")+m_growMeshName);

	HesperisIO::WriteMeshes(meshes, file);
}

MStatus HesperisCmd::printHelp()
{
	MGlobal::displayInfo(MString("To use hesperis cmd:")
		+MString("\n select group of curves to export")
		+MString("\n hesperis -w filename -gm fullPathToMesh")
        +MString("\n -gm or -growMesh is full path name to the transform of grow mesh"));
	return MS::kSuccess;
}
//:~