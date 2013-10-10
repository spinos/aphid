#include "CacheFile.h"
#include <AllHdf.h>
#include <HBase.h>

CacheFile::CacheFile() : HFile() {}
CacheFile::CacheFile(const char * name) : HFile(name) {}

bool CacheFile::create(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	setDocument(HObject::FileIO);

	return BaseFile::create(fileName);
}

bool CacheFile::open(const std::string & fileName)
{
    if(!FileExists(fileName)) {
		setLatestError(BaseFile::FileNotFound);
		return false;
	}
	
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	setDocument(HObject::FileIO);

	return BaseFile::open(fileName);
}

bool CacheFile::save()
{
    return true;
}

bool CacheFile::close()
{
	useDocument();
	std::map<std::string, HBase *>::iterator it;
	for(it = m_entries.begin(); it != m_entries.end(); ++it) {
		(*it).second->close();
		delete (*it).second;
	}
		
	return HFile::close();
}

void CacheFile::addEntry(const std::string & name)
{
	HBase * entry = new HBase(name);
	m_entries[name] = entry;
}

void CacheFile::addSliceVector3(const std::string & entryName, const std::string & sliceName)
{
	if(m_entries.find(entryName) == m_entries.end()) return;
	
	HBase * g = m_entries[entryName];
	
	if(!g->hasNamedData(sliceName.c_str()))
		g->addVector3Data(sliceName.c_str(), 1024);
}
#include <sstream>
void CacheFile::openSlice(const std::string & entryName, const std::string & sliceName)
{
	if(m_entries.find(entryName) == m_entries.end()) return;
	HBase * g = m_entries[entryName];
	
	if(!g->hasNamedData(sliceName.c_str())) return;
	
	VerticesHDataset * pset = new VerticesHDataset(sliceName.c_str());
	pset->open(g->fObjectId);
	
	std::stringstream sst;
	sst.str("");
	sst<<entryName<<"/"<<sliceName;
	
	std::cout<<"open "<<sst.str();
	m_slices[sst.str()] = pset;
}

void CacheFile::closeSlice(const std::string & entryName, const std::string & sliceName)
{
	std::stringstream sst;
	sst.str("");
	sst<<entryName<<"/"<<sliceName;
	if(m_slices.find(sst.str()) == m_slices.end()) return;
	
	std::cout<<"to close "<<sst.str();
	m_slices[sst.str()]->close();
}

void CacheFile::writeSliceVector3(const std::string & entryName, const std::string & sliceName, unsigned start, unsigned count, Vector3F * data)
{
	std::stringstream sst;
	sst.str("");
	sst<<entryName<<"/"<<sliceName;
	if(m_slices.find(sst.str()) == m_slices.end()) return;
	
	VerticesHDataset * p = (VerticesHDataset *)m_slices[sst.str()];
	p->setNumVertices(start + count);
	
	if(!p->hasEnoughSpace()) p->resize();
	
	HDataset::SelectPart pt;
	pt.start[0] = start;
	pt.count[0] = 1;
	pt.block[0] = count * 3;

	p->write((char *)data, &pt);
}
