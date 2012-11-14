//-*****************************************************************************
//
// Copyright (c) 2009-2011,
//  Sony Pictures Imageworks, Inc. and
//  Industrial Light & Magic, a division of Lucasfilm Entertainment Company Ltd.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Sony Pictures Imageworks, nor
// Industrial Light & Magic nor the names of their contributors may be used
// to endorse or promote products derived from this software without specific
// prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//-*****************************************************************************

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <ALFile.h>
#include <ALTransform.h>
#include <ALMesh.h>
#include <iostream>
#include <sstream>

//-*****************************************************************************
using namespace ::Alembic::AbcGeom;

static const std::string g_sep( ";" );



//-*****************************************************************************
// FORWARD
void visitProperties( ICompoundProperty, std::string & );

//-*****************************************************************************
template <class PROP>
void visitSimpleProperty( PROP iProp, const std::string &iIndent )
{
    std::string ptype = "ScalarProperty ";
    if ( iProp.isArray() ) { ptype = "ArrayProperty "; }


    std::string mdstring = "interpretation=";
    mdstring += iProp.getMetaData().get( "interpretation" );

    std::stringstream dtype;
    dtype << "datatype=";
    dtype << iProp.getDataType();

    mdstring += g_sep;

    mdstring += dtype.str();

    std::cout << iIndent << "  " << ptype << "name=" << iProp.getName()
              << g_sep << mdstring << g_sep << "numsamps="
              << iProp.getNumSamples() << std::endl;
}

//-*****************************************************************************
void visitCompoundProperty( ICompoundProperty iProp, std::string &ioIndent )
{
    std::string oldIndent = ioIndent;
    ioIndent += "  ";

    std::string interp = "schema=";
    interp += iProp.getMetaData().get( "schema" );

    std::cout << ioIndent << "CompoundProperty " << "name=" << iProp.getName()
              << g_sep << interp << std::endl;

    visitProperties( iProp, ioIndent );

    ioIndent = oldIndent;
}

//-*****************************************************************************
void visitProperties( ICompoundProperty iParent,
                      std::string &ioIndent )
{
    std::string oldIndent = ioIndent;
    for ( size_t i = 0 ; i < iParent.getNumProperties() ; i++ )
    {
        PropertyHeader header = iParent.getPropertyHeader( i );

        if ( header.isCompound() )
        {
            visitCompoundProperty( ICompoundProperty( iParent,
                                                      header.getName() ),
                                   ioIndent );
        }
        else if ( header.isScalar() )
        {
            visitSimpleProperty( IScalarProperty( iParent, header.getName() ),
                                 ioIndent );
        }
        else
        {
            assert( header.isArray() );
            visitSimpleProperty( IArrayProperty( iParent, header.getName() ),
                                 ioIndent );
        }
    }

    ioIndent = oldIndent;
}

//-*****************************************************************************
void visitObject( IObject iObj,
                  std::string iIndent )
{
    // Object has a name, a full name, some meta data,
    // and then it has a compound property full of properties.
    std::string path = iObj.getFullName();

    if ( path != "/" )
    {
        std::cout << "Object " << "name=" << path << std::endl;
    }

    // Get the properties.
    ICompoundProperty props = iObj.getProperties();
    visitProperties( props, iIndent );

    // now the child objects
    for ( size_t i = 0 ; i < iObj.getNumChildren() ; i++ )
    {
        visitObject( IObject( iObj, iObj.getChildHeader( i ).getName() ),
                     iIndent );
    }
}

void showTimeSampling(IArchive archive)
{
	std::cout<<"num time samplings "<<archive.getNumTimeSamplings()<<std::endl;
	AbcA::TimeSamplingPtr sampler = archive.getTimeSampling(0);
	std::cout<<"time sampling[0] "<<sampler->getSampleTime(0)<<std::endl;
}

ALTransform addAbcGroup(ALFile &file, const char *name, const char *term)
{
    OObject p;
    file.findParentOf(name, p);

    ALTransform res(p, term);
    res.addTranslate(3.0, 4.0, 5.0);
	/*group1.addRotate(0.5, 0.0, 0.0, 0);
	group1.addScale(2.0, 2.0, 2.0);
	group1.addScalePivot(0,0,0);
	group1.addScalePivotTranslate(0,0,0);
	group1.addRotatePivot(0,0,0);
	group1.addRotatePivotTranslate(0,0,0);*/
	res.write();
	return res;
}

void fillUV(float *uvs, unsigned *uvIds)
{
    //uvs = new float[8];
	uvs[0] = 0;
	uvs[1] = 0;
	uvs[2] = 1;
	uvs[3] = 0;
	uvs[4] = 1;
	uvs[5] = 1;
	uvs[6] = 0;
	uvs[7] = 1;

	//uvIds = new unsigned[6];
	uvIds[0] = 0;
	uvIds[1] = 1;
	uvIds[2] = 2;
	uvIds[3] = 4;
	uvIds[4] = 3;
	uvIds[5] = 0;
}

void write(const char * filename)
{
    std::cout<<"write "<<filename<<"\n";
    
    ALFile afile;
    
    afile.openAbc(filename);
	afile.addTimeSampling();
	afile.addTransform("|group1");
	
	ALTransform t = afile.lastTransform();
	t.addTranslate(0.0, 4.0, 5.0);
	t.write();
	
	afile.addTransform("|group1|group2");
	afile.addTransform("|group1|group2|group3");
	afile.addMesh("|group1|group2|group3|shape3");
	
	ALMesh shape3 = afile.lastMesh();
	
	const float vertices[12] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
	shape3.addP(vertices, 4);
	
	const unsigned indices[6] = {0, 1, 2, 2, 3, 0};
	shape3.addFaceConnection(indices, 6);
	
	const unsigned counts[2] = {3, 3};
	shape3.addFaceCount(counts, 2);
	
	float *uvs = 0;

	unsigned *uvIds = 0;

	//
	uvs = new float[8];
	

	uvIds = new unsigned[6];
	fillUV(uvs, uvIds);

	shape3.addUV(uvs, 4, uvIds, 6);
	
	shape3.write();
	//delete[] uvIds;
	//delete[] uvs;
	//ALTransform g0 = addAbcGroup(afile, "|group1", "group1");
	//ALTransform g1 = addAbcGroup(afile, "|group1|group2", "group2");
	//ALTransform g3 = addAbcGroup(afile, "|group1|group2|group3", "group3");
	//addAbcGroup(afile, "|group1|group3", "group3", p);
	/*OObject p;
	afile.findParentOf("|group1", p);
	ALTransform group1(p, "group1");
	
	OObject p1;
	afile.findParentOf("|group1|group2", p1);
	ALTransform group2(p1, "group2");*/

	//afile.findParentOf("|group1|meshShape", p);
	//ALMesh mesh(p, "meshShape");
	//mesh.write();
	std::cout<<"cleanup";
}

void read(const char * filename)
{
    std::cout<<"read "<<filename<<"\n";
	IArchive archive( Alembic::AbcCoreHDF5::ReadArchive(),
                          filename);
        if (archive)
        {
            std::cout  << "AbcEcho for " 
                       << Alembic::AbcCoreAbstract::GetLibraryVersion ()
                       << std::endl;;
        
            std::string appName;
            std::string libraryVersionString;
            uint32_t libraryVersion;
            std::string whenWritten;
            std::string userDescription;
            GetArchiveInfo (archive,
                            appName,
                            libraryVersionString,
                            libraryVersion,
                            whenWritten,
                            userDescription);
        
            if (appName != "")
            {
                std::cout << "  file written by: " << appName << std::endl;
                std::cout << "  using Alembic : " << libraryVersionString << std::endl;
                std::cout << "  written on : " << whenWritten << std::endl;
                std::cout << "  user description : " << userDescription << std::endl;
                std::cout << std::endl;
            }
            else
            {
                std::cout << filename << std::endl;
                std::cout << "  (file doesn't have any ArchiveInfo)" 
                          << std::endl;
                std::cout << std::endl;
            }
        }
		showTimeSampling(archive);
        visitObject( archive.getTop(), "" );
}

//-*****************************************************************************
//-*****************************************************************************
// DO IT.
//-*****************************************************************************
//-*****************************************************************************
int main( int argc, char *argv[] )
{
	std::cout<<"hello abc\n";
    if ( argc != 2 )
    {
        //std::cerr << "USAGE: " << argv[0] << " <AlembicArchive.abc>"
          //        << std::endl;
        //exit( -1 );
		write("./foo.abc");
		read("./foo.abc");
		exit(0);
    }
	
	read(argv[1]);

    return 0;
}
