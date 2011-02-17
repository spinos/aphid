#include <iostream>

#include "customer.h"
 
using namespace std;
 
void say_hello(const char* name) {
    cout << "Hello " <<  name << "!\n";
}
 
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
using namespace boost::python;

 
BOOST_PYTHON_MODULE(hello)
{
   def("say_hello", say_hello);
    
   class_<Customer>("Customer")
  		.def( init<std::string>() )        // Overloaded constructor version #1
		.def( "setName", &Customer::setName )
		.def( "getName", &Customer::getName )
		;
}
