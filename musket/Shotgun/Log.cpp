/*
 *  Log.cpp
 *  shotgunAPI
 *
 *  Created by jian zhang on 3/19/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Log.h"

namespace SG {
Log::Log() {}

void Log::print(const xmlrpc_c::paramList & params)
{
	for(unsigned i=0; i < params.size(); i++) {
		std::clog<<"param["<<i<<"]\n";
		printValues(params.getStruct(i));
	}
}

void Log::printValues(const std::map<std::string, xmlrpc_c::value> & values)
{
	std::map<std::string, xmlrpc_c::value>::const_iterator it = values.begin();
	for(; it != values.end(); ++it) {
		std::clog<<"  "<<it->first<<"<"<<xmlrpcValueTypeStr(it->second.type())<<">";
		printValue(it->second);
	}
}

void Log::printValues(const std::vector<xmlrpc_c::value> & values)
{
	std::vector<xmlrpc_c::value>::const_iterator it = values.begin();
	for(; it != values.end(); ++it) {
		std::clog<<"<"<<xmlrpcValueTypeStr(it->type())<<">";
		printValue(*it);
	}
}

void Log::printValue(const xmlrpc_c::value & value)
{
	std::clog<<" :";
	int vi;
	double vd;
	bool vb;
	time_t vt;
	std::string vs;
	std::map<std::string, xmlrpc_c::value> vstruct;
	std::vector<xmlrpc_c::value> va;
	switch (value.type())
    {
        case xmlrpc_c::value::TYPE_INT:
            fromXmlrpcValue(value, vi);
			std::clog<<" "<<vi<<"\n";
			break;
        case xmlrpc_c::value::TYPE_BOOLEAN:
            fromXmlrpcValue(value, vb);
			std::clog<<" "<<vb<<"\n";
			break;
        case xmlrpc_c::value::TYPE_DOUBLE:
            fromXmlrpcValue(value, vd);
			std::clog<<" "<<vd<<"\n";
			break;
        case xmlrpc_c::value::TYPE_DATETIME:
            fromXmlrpcValue(value, vt);
			std::clog<<" "<<vt<<"\n";
			break;
        case xmlrpc_c::value::TYPE_STRING:
            fromXmlrpcValue(value, vs);
			std::clog<<" "<<vs<<"\n";
			break;
		case xmlrpc_c::value::TYPE_STRUCT:
			fromXmlrpcValue(value, vstruct);
			std::clog<<" struct begin {\n";
			printValues(vstruct);
			std::clog<<"} struct end\n";
            break;
		case xmlrpc_c::value::TYPE_ARRAY:
			fromXmlrpcValue(value, va);
			std::clog<<" array begin {\n";
			printValues(va);
			std::clog<<"} array end\n";
            break;
		/*
        case xmlrpc_c::value::TYPE_BYTESTRING:
            return std::string("TYPE_BYTESTRING");
        case xmlrpc_c::value::TYPE_C_PTR:
            return std::string("TYPE_C_PTR");
        case xmlrpc_c::value::TYPE_NIL:
            return std::string("TYPE_NIL");
        case xmlrpc_c::value::TYPE_DEAD:*/
        default:
            std::clog<<" unknown\n";
    }
}
}