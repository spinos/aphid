#include <ALIFile.h>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <string>
#include <Alembic/Abc/All.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <ALITransform.h>
#include <ALIMesh.h>

ALIFile::ALIFile() {}
ALIFile::~ALIFile() 
{
    flush();
}

bool ALIFile::loadAbc(const char *filename)
{
    std::string abcName(filename);
    int found = abcName.rfind('.', abcName.size());
    if(found > 0)
        abcName.erase(found);
    abcName.append(".abc");

    m_archive = IArchive(Alembic::AbcCoreHDF5::ReadArchive(),
                                abcName);
    
    if (!m_archive.valid()) {
        std::cout<<"failed to open file "<<filename<<" to read\n";
        return 0;
    }
    
    std::cout<<"num time samplings "<<m_archive.getNumTimeSamplings()<<std::endl;
	AbcA::TimeSamplingPtr sampler = m_archive.getTimeSampling(0);
	std::cout<<"time sampling[0] "<<sampler->getSampleTime(0)<<std::endl;
    
	listObject(m_archive.getTop(), "");
    return 1;
}

void ALIFile::listObject( IObject iObj, std::string iIndent )
{
    // Object has a name, a full name, some meta data,
    // and then it has a compound property full of properties.
    std::string path = iObj.getFullName();

    if ( path != "/" ) {
        std::cout << "Object " << path << std::endl;
        if(Alembic::AbcGeom::IXform::matches(iObj.getHeader())) {
            std::cout<<"transform\n";
            Alembic::AbcGeom::IXform xform(iObj, Alembic::Abc::kWrapExisting);
            m_transform.push_back(new ALITransform(xform));
        }
        else if(Alembic::AbcGeom::IPolyMesh::matches(iObj.getHeader())) {
            std::cout<<"mesh\n";
            Alembic::AbcGeom::IPolyMesh mesh(iObj, Alembic::Abc::kWrapExisting);
            m_mesh.push_back(new ALIMesh(mesh));
        }
    }

    // Get the properties.
    //ICompoundProperty props = iObj.getProperties();
    //visitProperties( props, iIndent );

    // now the child objects
    for ( size_t i = 0 ; i < iObj.getNumChildren() ; i++ ) {
        listObject( IObject( iObj, iObj.getChildHeader( i ).getName() ),
                     iIndent );
    }
}

unsigned ALIFile::numTransform() const
{
    return m_transform.size();
}

unsigned ALIFile::numMesh() const
{
    return m_mesh.size();
}

void ALIFile::flush()
{
    std::vector<ALITransform *>::iterator itTransform;
    for(itTransform = m_transform.begin(); itTransform != m_transform.end(); itTransform++)
        delete *itTransform;
    
    std::vector<ALIMesh *>::iterator itMesh;
    for(itMesh = m_mesh.begin(); itMesh != m_mesh.end(); itMesh++)
        delete *itMesh;

    m_transform.clear();
    m_mesh.clear();
}

ALITransform *ALIFile::getTransform(unsigned idx)
{
    return m_transform[idx];
}
	
ALIMesh *ALIFile::getMesh(unsigned idx)
{
    return m_mesh[idx];
}
//~:
