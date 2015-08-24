#include <maya/MPxCommand.h>
#include <maya/MDagPathArray.h>
#include <maya/MSyntax.h>
#include <maya/MArgList.h>
#include <maya/MString.h>
#include <maya/MSelectionList.h>
#include <Vector3F.h>
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
    MStatus writeSelectedCurve(const MSelectionList & selList);
    MStatus writeSelectedMesh(const MSelectionList & selList);
    MStatus deformSelected();
	MStatus attachSelected(const Vector3F & offsetV);
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
    
    enum HandleType {
        HTAll = 0,
        HTCurve = 1,
        HTMesh = 2
    };
    
	IOMode m_ioMode;
    HandleType m_handelType;
	MString m_fileName;
	MString m_growMeshName;
};
