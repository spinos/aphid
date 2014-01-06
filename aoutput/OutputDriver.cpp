//////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2011, Image Engine Design Inc. All rights reserved.
//  Copyright (c) 2012, John Haddon. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are
//  met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of Image Engine Design nor the names of any
//       other contributors to this software may be used to endorse or
//       promote products derived from this software without specific prior
//       written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
//  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//////////////////////////////////////////////////////////////////////////
#include <ai.h>
#include <ai_drivers.h>
#include <iostream>
#include <sstream>
#include <boost/asio.hpp>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace boost::posix_time;
using boost::asio::ip::tcp;
#define PACKAGESIZE 1024
#define NUMFLOATPERPACKAGE 256
class DisplayDriver
{
public:
    DisplayDriver() {}
    void setChannelNames(std::vector<std::string> channelNames) {
        m_channels = channelNames;
    }
    unsigned numChannels() const {
        return m_channels.size();
    }
private:
    std::vector<std::string> m_channels;
};

typedef DisplayDriver* DisplayDriverPtr;

static AtVoid driverParameters( AtList *params, AtMetaDataStore *metaData )
{
	AiParameterSTR( "driverType", "" );

	// we need to specify this metadata to keep MtoA happy.
	AiMetaDataSetStr( metaData, 0, "maya.attr_prefix", "" );
	AiMetaDataSetStr( metaData, 0, "maya.translator", "ie" );
}

static AtVoid driverInitialize( AtNode *node, AtParamValue *parameters )
{
	AiDriverInitialize( node, true, new DisplayDriverPtr );
}

static AtVoid driverUpdate( AtNode *node, AtParamValue *parameters )
{
}

static bool driverSupportsPixelType( const AtNode *node, AtByte pixelType )
{
	switch( pixelType )
	{
		case AI_TYPE_RGB :
		case AI_TYPE_RGBA :
		case AI_TYPE_POINT :
			return true;
		default:
			return false;
	}
}

static const char **driverExtension()
{
   return 0;
}

static AtVoid driverOpen( AtNode *node, struct AtOutputIterator *iterator, AtBBox2 displayWindow, AtBBox2 dataWindow, int bucketSize )
{	
    std::vector<std::string> channelNames;
	std::stringstream sst;
	    
	const char *name = 0;
	int pixelType = 0;
	while( AiOutputIteratorGetNext( iterator, &name, &pixelType, 0 ) ) {
	    sst.str("");
	    sst<<name;
		std::string namePrefix;
		if( sst.str() == "RGB" || sst.str() == "RGBA" )
		{
			namePrefix = sst.str() + ".";
			AiMsgInfo( sst.str().c_str());
		}
		
		switch( pixelType )
		{
			case AI_TYPE_RGBA :
				channelNames.push_back( namePrefix + "R" );
				channelNames.push_back( namePrefix + "G" );
				channelNames.push_back( namePrefix + "B" );
				channelNames.push_back( namePrefix + "A" );
				break;
			default :
				break;
		}
	}
	
	sst.str("");
	sst<<"displayWindow "<<displayWindow.minx<<" "<<displayWindow.maxx<<" "<<displayWindow.miny<<" "<<displayWindow.maxy;
	AiMsgInfo(sst.str().c_str());
	
	sst.str("");
	sst<<"dataWindow "<<dataWindow.minx<<" "<<dataWindow.maxx<<" "<<dataWindow.miny<<" "<<dataWindow.maxy;
	AiMsgInfo(sst.str().c_str());
	
	sst.str("");
	sst<<"bucketSize "<<bucketSize;
	AiMsgInfo(sst.str().c_str());
	
	sst.str("");
	sst<<"nchannels "<<channelNames.size();
	AiMsgInfo(sst.str().c_str());
	/*
	/// \todo Make Convert.h
	Box2i cortexDisplayWindow(
		V2i( displayWindow.minx, displayWindow.miny ),
		V2i( displayWindow.maxx, displayWindow.maxy )
	);

	Box2i cortexDataWindow(
		V2i( dataWindow.minx, dataWindow.miny ),
		V2i( dataWindow.maxx, dataWindow.maxy )
	);
		
	CompoundDataPtr parameters = new CompoundData();
	ToArnoldConverter::getParameters( node, parameters->writable() );	
*/
	//const char *driverType = AiNodeGetStr( node, "driverType" );
	
	DisplayDriverPtr *driver = (DisplayDriverPtr *)AiDriverGetLocalData( node );
	*driver = new DisplayDriver;
	(*driver)->setChannelNames(channelNames);
	/*
	try
	{
		*driver = IECore::DisplayDriver::create( driverType, cortexDisplayWindow, cortexDataWindow, channelNames, parameters );
	}
	catch( const std::exception &e )
	{
		// we have to catch and report exceptions because letting them out into pure c land
		// just causes aborts.
		msg( Msg::Error, "ieOutputDriver:driverOpen", e.what() );
	}*/
}

static AtVoid driverPrepareBucket( AtNode *node, AtInt x, AtInt y, AtInt sx, AtInt sy, AtInt tId )
{
}

static AtVoid driverWriteBucket( AtNode *node, struct AtOutputIterator *iterator, struct AtAOVSampleIterator *sampleIterator, AtInt x, AtInt y, AtInt sx, AtInt sy ) 
{
	DisplayDriverPtr *driver = (DisplayDriverPtr *)AiDriverGetLocalData( node );
	if( !*driver ) return;
	if((*driver)->numChannels() != 4) return;
	
	std::stringstream sst;
	sst<<"bucketCoornidate "<<x<<" "<<x + sx - 1<<" "<<y<<" "<<y + sy - 1;
	AiMsgInfo(sst.str().c_str());
	
	const int numOutputChannels = (*driver)->numChannels();
    
	std::vector<float> interleavedData;
	interleavedData.resize( sx * sy * numOutputChannels );

	int pixelType = 0;
	const AtVoid *bucketData;
	int outChannelOffset = 0;
	while( AiOutputIteratorGetNext( iterator, 0, &pixelType, &bucketData ) )
	{
		int numChannels = 0;
		switch( pixelType )
		{
			case AI_TYPE_RGB :
			case AI_TYPE_VECTOR :
			case AI_TYPE_POINT :
				numChannels = 3;
				break;
			case AI_TYPE_RGBA :
				numChannels = 4;
				break;
			case AI_TYPE_FLOAT :
				numChannels = 1;
				break;
		}
		
		if(numChannels == 4) {
            for( int c = 0; c < numChannels; c++ )
            {
                float *in = (float *)(bucketData) + c;
                float *out = &(interleavedData[0]) + outChannelOffset;
                for( int j = 0; j < sy; j++ )
                {
                    for( int i = 0; i < sx; i++ )
                    {
                        *out = *in;
                        out += numOutputChannels;
                        in += numChannels;
                    }
                }
                outChannelOffset += 1;
            }
            break;
		}
	}

	char dataPackage[PACKAGESIZE];
        
    int * rect = (int *)dataPackage;			
    rect[2] = y;
    rect[3] = y + sy - 1;
    rect[0] = x;
    rect[1] = x + sx - 1;

	const unsigned npix = (rect[1] - rect[0] + 1) * (rect[3] - rect[2] + 1);
    int npackage = npix * 16 / PACKAGESIZE;
    if((npix * 16) % PACKAGESIZE > 0) npackage++;
	
	try {
        boost::asio::io_service io_service;
        tcp::resolver resolver(io_service);
        tcp::resolver::query query(tcp::v4(), "localhost", "7879");
        tcp::resolver::iterator sockIterator = resolver.resolve(query);
        tcp::socket s(io_service);
        s.connect(*sockIterator);
    
        boost::asio::write(s, boost::asio::buffer(dataPackage, 16));
                    
        boost::array<char, 32> buf;
        boost::system::error_code error;
                    
        size_t reply_length = s.read_some(boost::asio::buffer(buf), error);
        
        float *color = (float *)dataPackage;
        unsigned packageStart = 0;
        for(int i=0; i < npackage; i++) {
            for(int i = 0; i < NUMFLOATPERPACKAGE; i++) {
                if(packageStart + i == npix * 4) break;
                color[i] = interleavedData[packageStart + i];
            }
            packageStart += NUMFLOATPERPACKAGE;
            boost::asio::write(s, boost::asio::buffer(dataPackage, PACKAGESIZE));
            reply_length = s.read_some(boost::asio::buffer(buf), error);
        }
        dataPackage[0] = '\n';
        boost::asio::write(s, boost::asio::buffer(dataPackage, PACKAGESIZE));
				
        reply_length = s.read_some(boost::asio::buffer(buf), error);

        s.close();
        //boost::asio::deadline_timer t(io_service);
        //t.expires_from_now(boost::posix_time::seconds(1));
		//t.wait();
    }
    catch (std::exception& e)
	{
		AiMsgInfo(e.what());
	}
	/*
	Box2i bucketBox(
		V2i( x, y ),
		V2i( x + sx - 1, y + sy - 1 )
	);
	try
	{
		(*driver)->imageData( bucketBox, &(interleavedData[0]), interleavedData.size() );
	}
	catch( const std::exception &e )
	{
		// we have to catch and report exceptions because letting them out into pure c land
		// just causes aborts.
		msg( Msg::Error, "ieOutputDriver:driverWriteBucket", e.what() );
	}*/
}

static AtVoid driverClose( AtNode *node, struct AtOutputIterator *iterator )
{
	DisplayDriverPtr *driver = (DisplayDriverPtr *)AiDriverGetLocalData( node );
	if( *driver )
	{
		try
		{
			//(*driver)->imageClose(); 
		}
		catch( const std::exception &e )
		{
			// we have to catch and report exceptions because letting them out into pure c land
			// just causes aborts.
			//msg( Msg::Error, "ieOutputDriver:driverClose", e.what() );
		}
	}
}

static AtVoid driverFinish( AtNode *node )
{
	DisplayDriverPtr *driver = (DisplayDriverPtr *)AiDriverGetLocalData( node );
	delete driver;
	AiDriverDestroy( node );
}

AI_EXPORT_LIB AtBoolean NodeLoader( int i, AtNodeLib *node )
{
	if( i==0 )
	{		
		static AtCommonMethods commonMethods = { 
			driverParameters,
			driverInitialize,
			driverUpdate,
			driverFinish
		};
		static AtDriverNodeMethods driverMethods = {
			driverSupportsPixelType,
			driverExtension,
			driverOpen,
			driverPrepareBucket,
			driverWriteBucket,
			driverClose
		};
		static AtNodeMethods nodeMethods = {
			&commonMethods,
			&driverMethods
		};
		
		node->node_type = AI_NODE_DRIVER;
		node->output_type = AI_TYPE_NONE;
		node->name = "driver_foo";
		node->methods = &nodeMethods;
		sprintf( node->version, AI_VERSION );
		
		return true;
	}

	return false;
}
