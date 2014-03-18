# CMake project for shotgun C++ API
# 

cmake_minimum_required (VERSION 2.8)
project (shotgunAPI)

IF (WIN32)
  
ELSE()
  include_directories ("/usr/local/include")
ENDIF()

include_directories (".")

configure_file (
  "${PROJECT_BINARY_DIR}/shotgunAPIConfig.h.in"
  "${PROJECT_BINARY_DIR}/shotgunAPIConfig.h"
  )
include_directories("${PROJECT_BINARY_DIR}")


set (ALL_SRC "Shotgun/Asset.cpp"               
			"Shotgun/Delivery.cpp"           
			"Shotgun/Element.cpp"             
			"Shotgun/Entity.cpp"              
			"Shotgun/Group.cpp"               
			"Shotgun/Method.cpp"              
			"Shotgun/Note.cpp"                
			"Shotgun/NoteMixin.cpp"           
			"Shotgun/Playlist.cpp"            
			"Shotgun/PublishEvent.cpp"        
			"Shotgun/Review.cpp"              
			"Shotgun/ReviewItem.cpp"          
			"Shotgun/Sequence.cpp"            
			"Shotgun/Shot.cpp"                
			"Shotgun/Shotgun.cpp"             
			"Shotgun/Project.cpp"             
			"Shotgun/Task.cpp"                
			"Shotgun/TaskMixin.cpp"           
			"Shotgun/HumanUser.cpp"           
			"Shotgun/Version.cpp"             
			"Shotgun/CustomEntity.cpp"        
			"Shotgun/FilterBy.cpp"            
			"Shotgun/SortBy.cpp"              
			"Shotgun/List.cpp"                
			"Shotgun/Dict.cpp"                
			"Shotgun/Fields.cpp"              
			"Shotgun/types.cpp"               
			"Shotgun/exceptions.cpp"          
			"Shotgun/utils.cpp")
message("src files " ${ALL_SRC})

add_library(shotgunAPI STATIC ${ALL_SRC})

