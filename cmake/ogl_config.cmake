MESSAGE (" find ogl ")
find_package (OpenGL REQUIRED)
message ("opengl is " ${OPENGL_LIBRARIES})

IF (APPLE)

ELSEIF (UNIX)
find_package (GLEW)
message ("glew is " ${GLEW_LIBRARY})
ENDIF ()
