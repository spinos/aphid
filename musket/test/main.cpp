#include <iostream>
#include <stdexcept>

#include <Shotgun/types.h>
#include <Shotgun/exceptions.h>
#include <Shotgun/Shotgun.h>
#include <Shotgun/FilterBy.h>
#include <Shotgun/Dict.h>

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
        Shotgun sg;
        
        std::cout<<"\nsite URL "<<sg.serverURL()<<std::endl;
        std::clog<<"\ntest find one no field";
        SG::Entity * e = sg.find_one("Project", SG::FilterBy("id", "is", 65));
        std::cout<<"\nresult "<<e->str();
        SG::List resFld;
        std::clog<<"\ntest find one with empty field";
        e = sg.find_one("Project", SG::FilterBy("id", "is", 65), resFld);
        std::cout<<"\nreturn field "<<resFld;
        
        std::clog<<"\ntest find one with some field";
        resFld.clear();
        resFld.append("users");
        e = sg.find_one("Project", SG::FilterBy("id", "is", 65), resFld);
        std::cout<<"\nresult "<<e->str();

        resFld.clear();
        resFld.append("sg_type");
        resFld.append("users");
        e = sg.find_one("Project", SG::FilterBy("name", "is", "fool_zhang"), resFld);
        std::cout<<"\nresult "<<e->str();
        
        std::cout<<"\nchange sg_type";
        Dict retype("sg_type", "Misc");
        e = sg.update("Project", 273, retype);
        std::cout<<"\nreturn "<<e->str();
        
        e = sg.find_one("Project", SG::FilterBy("name", "is", "fool_zhang"), resFld);
        std::cout<<"\nresult "<<e->str();

        std::cout<<"\nupdate users";
        
        Dict usera("id", 102);
        usera.add("type", "HumanUser");
        Dict userb("id", 98);
        userb.add("type", "HumanUser");
        List uservalues;
        uservalues.append(usera);
        uservalues.append(userb);
        Dict reusers("users", uservalues);
        e = sg.update("Project", 273, reusers);
        
        e = sg.find_one("Project", SG::FilterBy("name", "is", "fool_zhang"), resFld);
        std::cout<<"\nresult "<<e->str();
        
        e = sg.find_one("CustomNonProjectEntity12");
        if(e)
            std::cout<<"\nresult "<<e->str();
        else
            std::cout<<"\nnot found CustomNonProjectEntity12\n";

        e = sg.find_one("HumanUser", SG::FilterBy("id", "is", 56), SG::List("sg_status_list").append("update_at"));
        std::cout<<e->str();
        SG::Entity ei = *e;
        std::cout<<" "<<ei["id"];
		std::cout<<sg.version();
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
