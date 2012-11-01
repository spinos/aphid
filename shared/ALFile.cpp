#include <ALFile.h>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <string>

#include <Alembic/AbcCoreHDF5/All.h>


ALFile::ALFile() {}
ALFile::~ALFile() {}

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

void ALFile::addTimeSampling()
{
	Alembic::AbcCoreAbstract::TimeSamplingPtr mShapeTime;
	Alembic::AbcCoreAbstract::TimeSamplingPtr mTransTime;
	
	mShapeTime.reset(new AbcA::TimeSampling());
	mTransTime.reset(new AbcA::TimeSampling());
	
	Alembic::Util::uint32_t mShapeTimeIndex = m_archive.addTimeSampling(*mShapeTime);
	
	Alembic::Util::uint32_t mTransTimeIndex = m_archive.addTimeSampling(*mTransTime);
	
	std::cout<<"shape time index "<<mShapeTimeIndex<<std::endl;
	std::cout<<"trans time index "<<mTransTimeIndex<<std::endl;
}

OObject ALFile::root()
{
    return m_archive.getTop();
}

void ALFile::splitNames(const std::string &fullPath, std::vector<std::string> &paths)
{
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|/");
	tokenizer tokens(fullPath, sep);
	
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
        if(child.getName() == name)
            return 1;
    }
    return 0;
}

char ALFile::findParentOf(const std::string &fullPathName, OObject &dest)
{
	std::vector<std::string > paths;
	splitNames(fullPathName, paths);

	std::string terminalName = paths.back();

	if(paths.size() < 2) {
		dest = root();
		return 1;
	}

	std::vector<std::string >::const_iterator lastPos = paths.begin();
	lastPos += paths.size() - 1;
	
	paths.erase(lastPos);

	return findObject(paths, dest);
}
