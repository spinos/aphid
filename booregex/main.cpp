#include <iostream>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/regex.hpp>
using namespace std;
using namespace boost::filesystem;


int
main() 
  {
    boost::match_results<std::string::const_iterator> what;
    cout<<"find date pattern\n";

    static const boost::regex e0(".*\\d{1,2}-\\d{1,2}-\\d{4}.*");

    const std::string log_text("this is the log:\ndata 1-29-2011 change minor\ndata 12-11-2010 change significant");

    regex_match(log_text, what, e0, boost::match_extra );

    if(what[0].matched)
        cout<<log_text<<" is matched\n";
    else
        cout<<log_text<<" is no matched\n";

    std::string::const_iterator start, end;
    start = log_text.begin();
    end = log_text.end();

    static const boost::regex expression("\\d{1,2}-\\d{1,2}-\\d{4}");

    while(regex_search(start, end, what, expression, boost::match_extra))
       {

cout<<what[0]<<endl;
           start = what[0].second;

       }

          cout<<"ls .cpp and .o files\n";
  	  static const boost::regex e("(.*)\\.o|(.*)\\.cpp");



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
