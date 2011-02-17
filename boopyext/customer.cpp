#include "customer.h"

Customer::Customer()
{
	_name = "NotSet";
}

Customer::Customer( std::string name )
{
	_name = name;
}

void Customer::setName( std::string name )
{
	_name = name;
}

std::string Customer::getName()
{
	return _name;
}
