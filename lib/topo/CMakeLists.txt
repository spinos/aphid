file (GLOB Topo_SRC
    "../../shared/topo/*.h"
    "../../shared/topo/*.cpp"
)

ADD_LIBRARY (aphidTopology ${LIB_TYP}
${Topo_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
