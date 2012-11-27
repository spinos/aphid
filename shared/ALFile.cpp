#include <ALFile.h>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <string>
#include <Alembic/Abc/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <ALTransform.h>
#include <ALMesh.h>

ALFile::ALFile() {}
ALFile::~ALFile() 
{
	flush();
}

void ALFile::openAbc(const char *filename)
{
    std::string abcName(filename);
    int found = abcName.rfind('.', abcName.size());
    if(found > 0)
        abcName.erase(found);
    abcName.append(".abc");

    m_archive = CreateArchiveWithInfo(Alembic::AbcCoreHDF5::WriteArchive(),
                                abcName, "opium write test", "foo info",
            ErrorHandler::kThrowPolicy);
    
    if (!m_archive.valid())
        std::cout<<"failed to open file "<<filename<<" to write\n";
}

void ALFile::addTimeSampling(double startTime, double endTime, double secondsPerFrame)
{
	if(startTime >= endTime) {
	    std::cout<<"no range "<<std::endl;
	    
	    mTransTime.reset(new AbcA::TimeSampling());
	    mShapeTime = mTransTime;
	}
	else {
	    std::vector<double> transSamples;
	    transSamples.push_back(startTime * secondsPerFrame);
	    
	    mTransTime.reset(new AbcA::TimeSampling(AbcA::TimeSamplingType(
                static_cast<Alembic::Util::uint32_t>(transSamples.size()),
                secondsPerFrame), transSamples));

        std::vector<double> shapeSamples;
        shapeSamples.push_back(startTime * secondsPerFrame);
        mShapeTime.reset(new AbcA::TimeSampling(AbcA::TimeSamplingType(
                secondsPerFrame), shapeSamples));
	}
	
	Alembic::Util::uint32_t mTransTimeIndex = m_archive.addTimeSampling(*mTransTime);
	Alembic::Util::uint32_t mShapeTimeIndex = m_archive.addTimeSampling(*mShapeTime);	
	
	std::cout<<"shape time index "<<mShapeTimeIndex<<std::endl;
	std::cout<<"trans time index "<<mTransTimeIndex<<std::endl;
}

OObject ALFile::root()
{
    return m_archive.getTop();
}

std::string ALFile::terminalName(const std::string &fullPathName)
{
    std::vector<std::string > paths;
	splitNames(fullPathName, paths);
	return paths.back();
}

void ALFile::splitNames(const std::string &fullPathName, std::vector<std::string> &paths)
{
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|/");
	tokenizer tokens(fullPathName, sep);
	
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		paths.push_back(*tok_iter);
	}
}

char ALFile::findObject(const std::vector<std::string> &path, OObject &dest)
{
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	
	OObject parent = root();

	for(std::vector<std::string >::const_iterator it = path.begin();
		it != path.end(); it++)
	{
		std::string r = *it;
		
		if(!findChildByName(parent, dest, r))
			return 0;
		    
		parent = dest;
	}

	return 1;
}

char ALFile::findObject(const std::string &fullPathName, OObject &dest)
{
	std::vector<std::string > paths;
	splitNames(fullPathName, paths);
	return findObject(paths, dest);
}

char ALFile::findChildByName(OObject &parent, OObject &child, const std::string &name)
{
    for ( size_t i = 0 ; i < parent.getNumChildren() ; i++ ) {
		child = parent.getChild(i);
		if(!child.valid()) {
		    std::cout<<"not valid\n";
		}
        else if(child.getName() == name)
            return 1;
    }
    return 0;
}

char ALFile::findParentOf(const std::string &fullPathName, OObject &dest)
{
	std::vector<std::string > paths;
	splitNames(fullPathName, paths);

	if(paths.size() < 2) {
		dest = root();
		return 1;
	}

	std::vector<std::string >::iterator lastPos = paths.begin();
	lastPos += paths.size() - 1;
	
	paths.erase(lastPos);

	return findObject(paths, dest);
}

bool ALFile::addTransform(const std::string &fullPathName)
{
    OObject p;
    if(!findParentOf(fullPathName, p))
        return 0;

    std::string name = terminalName(fullPathName);
	Alembic::AbcGeom::OXform obj(p, name, mTransTime);
    m_transform.push_back(new ALTransform(obj));
	return 1;
}

ALTransform *ALFile::lastTransform()
{
    return m_transform.back();
}

bool ALFile::addMesh(const std::string &fullPathName)
{
	OObject p;
    if(!findParentOf(fullPathName, p))
        return 0;
		
	std::string name = terminalName(fullPathName);
	Alembic::AbcGeom::OPolyMesh obj(p, name, mShapeTime);
    m_mesh.push_back(new ALMesh(obj));
	return 1;
}

ALMesh *ALFile::lastMesh()
{
	return m_mesh.back();
}

void ALFile::flush()
{
    std::vector<ALTransform *>::iterator itTransform;
    for(itTransform = m_transform.begin(); itTransform != m_transform.end(); itTransform++)
        delete *itTransform;
    
    std::vector<ALMesh *>::iterator itMesh;
    for(itMesh = m_mesh.begin(); itMesh != m_mesh.end(); itMesh++)
        delete *itMesh;

    m_transform.clear();
    m_mesh.clear();
}

void ALFile::begin()
{
    m_currentTransformIdx = 0;
	m_currentMeshIdx = 0;
}

ALTransform *ALFile::getTransform()
{
    return m_transform[m_currentTransformIdx];
}

ALMesh *ALFile::getMesh()
{
    return m_mesh[m_currentMeshIdx];
}

void ALFile::nextTransform()
{
    m_currentTransformIdx++;
}

void ALFile::nextMesh()
{
    m_currentMeshIdx++;
}
//:~
