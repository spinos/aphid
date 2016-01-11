#include <maya/MPlug.h>
#include <maya/MPxNode.h>
#include <maya/MDataBlock.h>
#include <maya/MObject.h> 
#include <maya/MFnNumericData.h>
#include <maya/MString.h>
#include <EnvVar.h>
#include <H5Holder.h>
#include <HOocArray.h>
#include <HBase.h>
#include <AHelper.h>

class H5AttribNode : public MPxNode, public EnvVar, public H5Holder
{
public:
						H5AttribNode();
	virtual				~H5AttribNode(); 

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	static  void*		creator();
	static  MStatus		initialize();

    virtual MStatus connectionMade(const MPlug &plug, const MPlug &otherPlug, bool asSrc);
	
public:
	static  	MObject		input;
	static  MObject aframe;
	static MObject aminframe;
	static MObject amaxframe;
	static MObject abyteAttrName;
	static  	MObject 	outByte;
	static MObject ashortAttrName;
	static  	MObject 	outShort;
	static MObject aintAttrName;
	static  	MObject 	outInt;
	static MObject afloatAttrName;
	static  	MObject 	outFloat;
	static MObject adoubleAttrName;
	static  	MObject 	outDouble;
	static MObject aboolAttrName;
	static  	MObject 	outBool;
	
	static	MTypeId		id;
	
private:
	static void createNameValueAttr(MObject & nameAttr, MObject & valueAttr,
						const MString & name1L, const MString & name1S, 
						const MString & name2L, const MString & name2S, 
						MFnNumericData::Type valueTyp);
						
	std::string getAttrNameInArray(MDataBlock& data, const MObject & attr, 
						unsigned idx, MStatus * stat) const;
						
	MDataHandle getHandleInArray(MDataBlock& data, const MObject & attr, 
						unsigned idx, MStatus * stat) const;
						
	std::map<std::string, HObject *> m_mappedAttribDatas;
	
	template<typename Td>
	Td * getDataStorage(HBase * grp, const std::string & attrName, bool & stat)
	{
		Td * d = NULL;
		if(m_mappedAttribDatas.find(attrName) == m_mappedAttribDatas.end() ) {
			d = grp->openDataStorage<Td>(".bake", stat);
			if(stat) m_mappedAttribDatas[attrName] = d;
			else AHelper::Info<std::string>("H5AttribNode cannot open data ", attrName);
		}
		else {
			d = static_cast<Td *>(m_mappedAttribDatas[attrName]);
			stat = true;
		}
		return d;
	}
	
	template<typename Td, typename Tv>
	bool readData(const std::string & attrName, SampleFrame * sampler, Tv & result)
	{
		HBase grp(attrName);
		bool stat;
		Td * d = getDataStorage<Td>(&grp, attrName, stat);
		if(stat) {
			Tv a, b;
			d->readElement((char *)&a, sampler->sampleOfset0() );
			if(sampler->m_weights[0] > .99f) {
				result = a;
			}
			else {
				d->readElement((char *)&b, sampler->sampleOfset1() );
				result = a * sampler->m_weights[0] + b * sampler->m_weights[1];
			}
		}
		
		grp.close();
		
		return stat;
	}
	
};


