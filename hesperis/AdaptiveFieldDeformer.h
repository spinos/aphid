#ifndef _AdaptiveFieldDeformer
#define _AdaptiveFieldDeformer

#include <maya/MPxNode.h>
#include <maya/MTypeId.h>
#include <maya/MPxDeformerNode.h>
#include <vector>
#include <MultiPlaybackFile.h>
using namespace std;
 
class AdaptiveFieldDeformer : public MPxDeformerNode
{
public:
						AdaptiveFieldDeformer();
	virtual				~AdaptiveFieldDeformer(); 

	virtual MStatus		deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex);

	static  void*		creator();
	static  MStatus		initialize();

	static	MTypeId		id;
	static  MObject aframe;
	static  MObject aminframe;
	static  MObject amaxframe;
	static  MObject acachename;
	static  MObject areadloc;
	static  MObject asubframe;
	
private:
	bool openFieldFile(const std::string & name);
	void setP(float env, float *p, MItGeometry& iter);
	char readFrame(float *data, int count, int frame, int sample);
	void mixFrames(float *p, float *p1, int count, float weight0, float weights1);
private:
	static MultiPlaybackFile AvailableFieldFiles;
	std::string m_lastFilename;
};

#endif
