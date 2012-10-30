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
            Alembic::Abc::ErrorHandler::kThrowPolicy);
    
    if (!m_archive.valid())
        std::cout<<"write failed\n";
}

Alembic::Abc::OObject ALFile::root()
{
    return m_archive.getTop();
}

char ALFile::object(const std::string &objectPath, Alembic::Abc::OObject &dest)
{
    std::cout<<"obj path "<<objectPath<<"\n";
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	
	std::string combined(objectPath);
	tokenizer tokens(combined, sep);
	
	Alembic::Abc::OObject parent = m_archive.getTop();
	Alembic::Abc::OObject child;
	std::string r;
	char found = 0;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		std::cout<<"try to find "<<r<<" within "<<parent.getName()<<"\n";
		
		found = findChildByName(parent, child, r);
		
		if(found) std::cout<<"found "<<r<<std::endl;
		
		tokenizer::iterator last = tok_iter;
		last++;
		
		if(last != tokens.end()) {
		    
		    if(!found)
		        return 0;
		    
		    parent = child;
		}
		
		
	}
	if(!found) {
	    dest = Alembic::Abc::OObject(parent, r);
	    std::cout<<"created "<<dest.getName()<<"\n";
	    
	}
	return found;
}

char ALFile::findChildByName(Alembic::Abc::OObject &parent, Alembic::Abc::OObject &child, const std::string &name)
{
    std::cout<<"child count "<<parent.getNumChildren()<<"\n";
    for ( size_t i = 0 ; i < parent.getNumChildren() ; i++ ) {
        std::cout<<" child "<<child.getName();
        child = parent.getChild(i);
        if(child.getName() == name)
            return 1;
    }
    return 0;
}
