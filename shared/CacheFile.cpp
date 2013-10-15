#include "CacheFile.h"
#include <AllHdf.h>
#include <HBase.h>

CacheFile::CacheFile() : HFile() {}
CacheFile::CacheFile(const char * name) : HFile(name) {}

void CacheFile::doClose()
{
	useDocument();
	std::map<std::string, HBase *>::iterator it;
	for(it = m_entries.begin(); it != m_entries.end(); ++it) {
	    if((*it).second) {
	        (*it).second->close();
	        delete (*it).second;
	    }
	}
		
	HFile::doClose();
}

HBase * CacheFile::getNamedEntry(const std::string & name)
{
    std::map<std::string, HBase *>::iterator it = m_entries.find(name);
    if(it == m_entries.end()) return 0;
    return (*it).second;
}

void CacheFile::openEntry(const std::string & name)
{
	HBase * entry = new HBase(name);
	m_entries[name] = entry;
}

void CacheFile::closeEntry(const std::string & name)
{
    HBase * p = getNamedEntry(name);
    if(p == 0) return;
	p->close();
	delete p;
	m_entries[name] = 0;
}

bool CacheFile::openSlice(const std::string & entryName, const std::string & sliceName)
{
	HBase * g = getNamedEntry(entryName);
	if(g == 0) return false;
	
	if(!g->hasNamedData(sliceName.c_str()))
	    g->addVector3Data(sliceName.c_str(), 2048);
	
	VerticesHDataset * pset = new VerticesHDataset(sliceName.c_str());
	pset->open(g->fObjectId);
	
	m_slices[HObject::FullPath(entryName, sliceName)] = pset;
	
	return true;
}

void CacheFile::closeSlice(const std::string & entryName, const std::string & sliceName)
{
	const std::string slicePath = HObject::FullPath(entryName, sliceName);
	
	if(m_slices.find(slicePath) == m_slices.end()) return;
	
	m_slices[slicePath]->close();
}

void CacheFile::saveEntrySize(const std::string & entryName, unsigned size)
{
    HBase * g = getNamedEntry(entryName);
	if(g == 0) return;
	
	if(!g->hasNamedAttr(".size"))
	    g->addIntAttr(".size");
	
	int x = size;
	g->writeIntAttr(".size", &x);
}

unsigned CacheFile::entrySize(const std::string & entryName)
{
    HBase * g = getNamedEntry(entryName);
	if(g == 0) return 0;
	
	if(!g->hasNamedAttr(".size"))
	    return 0;
	
	int x;
	g->readIntAttr(".size", &x);
	return x;
}

void CacheFile::writeSliceVector3(const std::string & entryName, const std::string & sliceName, unsigned start, unsigned count, Vector3F * data)
{
	const std::string slicePath = HObject::FullPath(entryName, sliceName);
	if(m_slices.find(slicePath) == m_slices.end()) return;
	
	VerticesHDataset * p = (VerticesHDataset *)m_slices[slicePath];
	p->setNumVertices(start + count);
	
	if(!p->hasEnoughSpace()) p->resize();
	
	HDataset::SelectPart pt;
	pt.start[0] = start * 3;
	pt.count[0] = 1;
	pt.block[0] = count * 3;

	p->write((char *)data, &pt);
}

void CacheFile::readSliceVector3(const std::string & entryName, const std::string & sliceName, unsigned start, unsigned count, Vector3F * data)
{
	const std::string slicePath = HObject::FullPath(entryName, sliceName);
	if(m_slices.find(slicePath) == m_slices.end()) return;
	
	VerticesHDataset * p = (VerticesHDataset *)m_slices[slicePath];
	
	unsigned maxcount = isCached(entryName, sliceName);
	
	if(maxcount < start) return;
	
	p->setNumVertices(maxcount);
	
	unsigned c = count;
	if(c > maxcount - start) c = maxcount - start;
	
	HDataset::SelectPart pt;
	pt.start[0] = start * 3;
	pt.count[0] = 1;
	pt.block[0] = c * 3;
	
	p->read((char *)data, &pt);
}

void CacheFile::setCached(const std::string & entryName, const std::string & sliceName, unsigned size)
{
	m_cachedSlices[HObject::FullPath(entryName, sliceName)] = size;
}

unsigned CacheFile::isCached(const std::string & entryName, const std::string & sliceName)
{
	const std::string slicePath = HObject::FullPath(entryName, sliceName);
	if(m_cachedSlices.find(slicePath) == m_cachedSlices.end()) return 0;
	return m_cachedSlices[slicePath];
}

void CacheFile::clearCached()
{
	m_cachedSlices.clear();
}

unsigned CacheFile::numCachedSlices(const std::string & entryName) const
{
    unsigned res = 0;
    std::map<std::string, unsigned>::const_iterator it;
	for(it = m_cachedSlices.begin(); it != m_cachedSlices.end(); ++it) {
	    if((*it).first.find(entryName, 0) == 0) 
	        res++;
	}
	
	return res;
}

unsigned CacheFile::cacheSliceNames(const std::string & entryName, std::vector<std::string> & dst) const
{
	unsigned res = 0;
    std::map<std::string, unsigned>::const_iterator it;
	for(it = m_cachedSlices.begin(); it != m_cachedSlices.end(); ++it) {
	    if((*it).first.find(entryName, 0) == 0) {
			dst.push_back((*it).first);
	        res++;
		}
	}
	
	return res;
}

void CacheFile::setBounding(const std::string & name, const BoundingBox & box)
{
	useDocument();
	HBase b("/bbox");
	if(!b.hasNamedAttr(name.c_str()))
		b.addFloatAttr(name.c_str(), 6);
		
	b.writeFloatAttr(name.c_str(), (float *)&box);
	b.close();
}

void CacheFile::getBounding(const std::string & name, BoundingBox & box)
{
	useDocument();
	HBase b("/bbox");
	if(b.hasNamedAttr(name.c_str()))
		b.readFloatAttr(name.c_str(), (float *)&box);
	b.close();
}