#include <iostream>
#include <stdexcept>

#include <Shotgun/types.h>
#include <Shotgun/exceptions.h>
#include <Shotgun/Shotgun.h>
#include <Shotgun/FilterBy.h>
#include <Shotgun/Dict.h>
#include <Shotgun/Fields.h>

using namespace SG;

int main( int argc, char **argv )
{
    std::string shotgunURL(SG_DEFAULT_URL);
    std::string authName(SG_AUTHENTICATION_NAME);
    std::string authKey(SG_AUTHENTICATION_KEY);
	
	std::clog<<"URL    "<<shotgunURL<<"\n";
	std::clog<<"author "<<authName<<"\n";
	std::clog<<"key    "<<authKey<<"\n";

    try
    {
        Shotgun sg(shotgunURL, authName, authKey);

        ProjectPtrs projects = sg.findEntities<Project>();
        std::cout << "\nlist all projects\n";
        for( size_t p = 0; p < projects.size(); ++p )
        {
            std::cout << *(projects[p]) << std::endl;
            //std::cout << projects[p]->getAttrValueAsInt("id") << std::endl;
            delete projects[p];
        }
		
		Project *project = sg.findEntity<Project>(FilterBy("id", "is", 65));
        std::cout << "\nfind a project";
        std::cout << *project << std::endl;
		delete project;
    }
    catch (const SgError & e)
    {
        std::cerr << "SgError: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
