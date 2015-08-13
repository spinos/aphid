#include <maya/MPxCommand.h>
#include <maya/MDagPathArray.h>
#include <maya/MSyntax.h>
#include <maya/MArgList.h>
#include <maya/MString.h>
#include <maya/MSelectionList.h>
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
	MStatus writeSelected(const MSelectionList & selList);
    MStatus deformSelected();
	MStatus attachSelected();
    void writeMesh(HesperisFile * file);
	MStatus printHelp();
	void testTransform();
private:
	enum IOMode {
		IOUnknown = 0,
		IOWrite = 1,
		IORead = 2,
        IOFieldDeform = 3,
		IOHelp = 4
	};
	IOMode m_ioMode;
	MString m_fileName;
	MString m_growMeshName;
};
