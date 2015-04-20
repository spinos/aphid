#include "HesperisCmd.h"
#include <maya/MGlobal.h>
#include <maya/MSelectionList.h>
#include <maya/MItSelectionList.h>
#include <maya/MItDag.h>
#include <maya/MArgDatabase.h>
#include "HesperisIO.h"

void *HesperisCmd::creator()
{ return new HesperisCmd; }

MSyntax HesperisCmd::newSyntax() 
{
	MSyntax syntax;

	syntax.addFlag("-w", "-write", MSyntax::kString);
	syntax.enableQuery(false);
	syntax.enableEdit(false);

	return syntax;
}

void HesperisCmd::getCurves(const MDagPath & root, MDagPathArray & dst)
{
	MStatus stat;
	MItDag iter;
	iter.reset(root, MItDag::kDepthFirst, MFn::kNurbsCurve);
	for(; !iter.isDone(); iter.next()) {								
		MDagPath apath;		
		iter.getPath( apath );
		dst.append(apath);
	}
}

MStatus HesperisCmd::parseArgs ( const MArgList& args )
{
	m_ioMode = IOUnknown;
	m_fileName = "";
	
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
	
	if(m_ioMode == IOUnknown) {
		MGlobal::displayInfo(" no valid arguments are set, use -w filename");
		return MS::kFailure;
	}
	
	return MS::kSuccess;
}

MStatus HesperisCmd::doIt(const MArgList &args)
{
	MStatus status = parseArgs( args );
	
	if( status != MS::kSuccess ) return status;
	
	MSelectionList selList;
    MGlobal::getActiveSelectionList(selList);
    
	if(selList.length() < 1) {
		MGlobal::displayInfo(" Empty selction!");
		return MS::kSuccess;
	}
	
	MItSelectionList iter( selList );
	
	MDagPathArray curves;
	
	for(; !iter.isDone(); iter.next()) {								
		MDagPath apath;		
		iter.getDagPath( apath );
		getCurves(apath, curves);	
	}
	
	if(curves.length() < 1) {
		MGlobal::displayInfo(" Zero curve selction!");
		return MS::kSuccess;
	}
	
	HesperisIO::WriteCurves(curves, m_fileName.asChar());
	
    return MS::kSuccess;
}


