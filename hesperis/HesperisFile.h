/*
 *  HesperisFile.h
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <HFile.h>
#include <string>
#include <map>
class CurveGroup;
class BaseBuffer;
struct TetrahedronMeshData;

class HesperisFile : public HFile {
public:
	enum ReadComponent {
		RNone = 0,
		RCurve = 1,
		RTetra = 2
	};

	enum WriteComponent {
		WCurve = 0,
		WTetra = 1
	};
	
	HesperisFile();
	HesperisFile(const char * name);
	virtual ~HesperisFile();
	
	void setReadComponent(ReadComponent comp);
	void setWriteComponent(WriteComponent comp);
	void addCurve(const std::string & name, CurveGroup * data);
	void addTetrahedron(const std::string & name, TetrahedronMeshData * data);
	virtual bool doWrite(const std::string & fileName);
	virtual bool doRead(const std::string & fileName);
protected:

private:
	bool writeCurve();
	bool writeTetrahedron();
	bool readCurve();
	bool readTetrahedron();
private:
	std::map<std::string, CurveGroup * > m_curves;
	std::map<std::string, TetrahedronMeshData * > m_terahedrons;
	ReadComponent m_readComp;
	WriteComponent m_writeComp;
};