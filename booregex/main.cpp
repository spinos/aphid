#include <iostream>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/regex.hpp>
using namespace std;
using namespace boost::filesystem;


int
main() 
  {    
  	  static const boost::regex e("(.*)\\.o|(.*)\\.cpp");
  	  boost::match_results<std::string::const_iterator> what;
   
  	  path head_path("/Users/jianzhang/aphid/booregex");
		directory_iterator end_iter;
		for ( directory_iterator itdir( head_path );
			  itdir != end_iter;
			  ++itdir )
		{
			if ( is_regular_file( itdir->status() ) )
			{
				
				regex_match(itdir->path().filename(), what, e, boost::match_partial );
				if(what[0].matched){
					
					cout<<itdir->path().filename().c_str()<<endl;
				}
					
				
			}
		}
    return 0;
  };
