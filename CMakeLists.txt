cmake_minimum_required(VERSION 3.22)

project(schrodinger_3d)

set(CMAKE_CXX_STANDARD 20)

function(add_shaders TARGET)
    find_program(GLSLANGVALIDATOR glslangValidator)
    foreach (SHADER ${ARGN})
        set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/src/shader/${SHADER})
        set(current-output-path ${CMAKE_CURRENT_SOURCE_DIR}/res/bin/shader/${SHADER}.spv)

        # Add a custom command to compile GLSL to SPIR-V using glslangvalidator.
        get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
        file(MAKE_DIRECTORY ${current-output-dir})

        add_custom_command(
                OUTPUT ${current-output-path}
                COMMAND ${GLSLANGVALIDATOR} -V -o ${current-output-path} ${current-shader-path}
                DEPENDS ${current-shader-path}
                IMPLICIT_DEPENDS CXX ${current-shader-path}
                VERBATIM)

        # Make sure our build depends on this output.
        set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)
        target_sources(${TARGET} PRIVATE ${current-output-path})
    endforeach ()
endfunction()

set(FT_DISABLE_ZLIB ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BZIP2 ON CACHE BOOL "" FORCE)
set(FT_DISABLE_PNG ON CACHE BOOL "" FORCE)
set(FT_DISABLE_HARFBUZZ ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BROTLI ON CACHE BOOL "" FORCE)

add_subdirectory(external/freetype-2.13.2)
add_subdirectory(lib/mve)

set(SOURCE_FILES
        src/main.cpp
        src/logger.cpp
        src/camera.cpp
        src/util/fixed_loop.cpp
        src/text_pipeline.cpp
        src/wire_box_mesh.cpp
        src/simple_pipeline.cpp
)

set(LIBS
        glfw
        freetype
        mve)

set(LIB_INCLUDES
        external/spdlog-1.13.0/include
        external/thread-pool-4.1.0/include)

add_executable(${PROJECT_NAME})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_link_libraries(${PROJECT_NAME} -static-libgcc -static-libstdc++)
endif ()

target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${LIBS})

target_include_directories(${PROJECT_NAME} PRIVATE ${LIB_INCLUDES})

add_shaders(schrodinger_3d
        cloud.frag
        cloud.vert
        text.frag
        text.vert
        simple.vert
        simple.frag)

