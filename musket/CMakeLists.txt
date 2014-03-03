# basic CMake project for ai shader
# 

cmake_minimum_required (VERSION 2.8)
project (shotgunAPI)

IF (WIN32)
  #include_directories ("D:/usr/arnoldSDK/arnold4014/include")
  #SET (CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS_INIT} "/machine:X64")
ELSE()
  include_directories ("~/Library/shotguncpp/lib/" "/usr/local/include")
ENDIF()

include_directories (".")

set (SG_DEFAULT_URL "localhost")
set (SG_AUTHENTICATION_NAME "default")
set (SG_AUTHENTICATION_KEY "unknown")

configure_file (
  "${PROJECT_BINARY_DIR}/shotgunAPIConfig.h.in"
  "${PROJECT_BINARY_DIR}/shotgunAPIConfig.h"
  )
include_directories("${PROJECT_BINARY_DIR}")


set (ALL_SRC "Asset.cpp"               
			"Delivery.cpp"           
			"Element.cpp"             
			"Entity.cpp"              
			"Group.cpp"               
			"Method.cpp"              
			"Note.cpp"                
			"NoteMixin.cpp"           
			"Playlist.cpp"            
			"PublishEvent.cpp"        
			"Review.cpp"              
			"ReviewItem.cpp"          
			"Sequence.cpp"            
			"Shot.cpp"                
			"Shotgun.cpp"             
			"Project.cpp"             
			"Task.cpp"                
			"TaskMixin.cpp"           
			"HumanUser.cpp"           
			"Version.cpp"             
			"CustomEntity.cpp"        
			"FilterBy.cpp"            
			"SortBy.cpp"              
			"List.cpp"                
			"Dict.cpp"                
			"Fields.cpp"              
			"types.cpp"               
			"exceptions.cpp"          
			"utils.cpp")
message("src files " ${ALL_SRC})

add_library(shotgunAPI STATIC ${ALL_SRC})
# target_link_libraries (ExtendShaders "D:/usr/arnoldSDK/arnold4014/lib/ai.lib")
