#include <maya/MPxCommand.h>
#include <maya/MDagPathArray.h>
#include <maya/MSyntax.h>
#include <maya/MArgList.h>
#include <maya/MString.h>
class HesperisFile;
class HesperisCmd : public MPxCommand {
public:                                                                                                                 
		HesperisCmd() {};
        virtual MStatus doIt (const MArgList &);
        static void*    creator();
		static MSyntax newSyntax();
private:
	void pushCurves(const MDagPathArray & curves);
	MStatus parseArgs ( const MArgList& args );
	void writeMesh(HesperisFile * file);
	MStatus printHelp();
	void testTransform();
private:
	enum IOMode {
		IOUnknown = 0,
		IOWrite = 1,
		IORead = 2,
		IOHelp = 3
	};
	IOMode m_ioMode;
	MString m_fileName;
	MString m_growMeshName;
};
