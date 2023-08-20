project(DERAPA)
cmake_minimum_required(VERSION 3.10)
include(ExternalProject)
set(DERAPA_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(DERAPA_HEADER_DIR ${PROJECT_SOURCE_DIR}/header)

if(NOT CMAKE_BUILD_TYPE)
	# set(CMAKE_BUILD_TYPE RELEASE)
	# set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
	set(CMAKE_BUILD_TYPE DEBUG)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -march=native -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

add_subdirectory(${PROJECT_SOURCE_DIR}/resource/ittnotify)

# ########################## External Dependencies #############################
ExternalProject_Add(catch2
	URL https://github.com/catchorg/Catch2/archive/v2.9.2.tar.gz
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${DERAPA_BINARY_DIR}/deps -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
)

################################### Targets ####################################
add_executable(DERAPA
	"${DERAPA_SRC_DIR}/benchmark/derapa.cpp"
)

find_package(OpenMP)
find_package(mimalloc)
find_package(Boost REQUIRED COMPONENTS program_options)


if(OpenMP_CXX_FOUND)
	target_link_libraries(DERAPA PUBLIC OpenMP::OpenMP_CXX)
endif()

if(mimalloc_FOUND)
	target_link_libraries(DERAPA PUBLIC mimalloc)
	target_link_libraries(DERAPA PUBLIC mimalloc-static)
	target_link_libraries(DERAPA PUBLIC /usr/local/lib/mimalloc-2.0/mimalloc.o)
	set(ENV{LD_PRELOAD} /usr/local/lib/libmimalloc.so.2.0)
	set(ENV{MIMALLOC_PAGE_RESET} 0)
	set(ENV{HUGETLB_MORECORE} thp)
	set(ENV{MIMALLOC_LARGE_OS_PAGES} 1)
	set(ENV{MIMALLOC_RESERVE_HUGE_OS_PAGES} 16)
	set(ENV{MIMALLOC_SHOW_STATS} 1)
	set(ENV{MIMALLOC_VERBOSE} 1)
else()
	ExternalProject_Add(mimalloc
		URL https://github.com/microsoft/mimalloc/archive/refs/heads/master.zip
		CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${DERAPA_BINARY_DIR}/deps -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
	)
	add_dependencies(DERAPA mimalloc)
	target_include_directories(DERAPA SYSTEM PUBLIC ${DERAPA_BINARY_DIR}/deps/include/mimalloc-2.0/)
	target_link_libraries(DERAPA PUBLIC ${DERAPA_BINARY_DIR}/deps/lib/mimalloc-2.0/mimalloc.o)
	set(ENV{LD_PRELOAD} ${DERAPA_BINARY_DIR}/deps/lib/libmimalloc.so.2.0)
	set(ENV{MIMALLOC_PAGE_RESET} 0)
	set(ENV{HUGETLB_MORECORE} thp)
	set(ENV{MIMALLOC_LARGE_OS_PAGES} 1)
	set(ENV{MIMALLOC_RESERVE_HUGE_OS_PAGES} 16)
	set(ENV{MIMALLOC_SHOW_STATS} 1)
	set(ENV{MIMALLOC_VERBOSE} 1)
endif()

if(Boost_FOUND)
	target_link_libraries(DERAPA PUBLIC Boost::program_options)
endif()

set_property(TARGET DERAPA PROPERTY CXX_STANDARD 17)

# add_dependencies(DERAPA Jemalloc)
# target_link_libraries(DERAPA ${JEMALLOC_LIBRARIES})

# target_link_libraries(DERAPA PUBLIC -lpthread)
# target_link_libraries(DERAPA PUBLIC -Wl,--no-as-needed -ldl)

if(APPLE)
	target_link_libraries(DERAPA PUBLIC "-framework CoreFoundation")
endif(APPLE)

target_include_directories(DERAPA SYSTEM PUBLIC ${DERAPA_BINARY_DIR}/deps/include)
# target_include_directories(DERAPA SYSTEM PUBLIC ${JEMALLOC_INSTALL_DIR}/include)
# target_link_libraries(DERAPA PUBLIC -ljemalloc)

target_link_libraries(DERAPA PUBLIC ittnotify)

# ################################## Tests #####################################
add_executable(DERAPA_Tests
	"${PROJECT_SOURCE_DIR}/test/adaptiveShiftIndex_test.cpp"
)

find_package(OpenMP)
find_package(mimalloc)
find_package(Boost REQUIRED COMPONENTS program_options)

if(OpenMP_CXX_FOUND)
	target_link_libraries(DERAPA_Tests PUBLIC OpenMP::OpenMP_CXX)
endif()

if(mimalloc_FOUND)
	target_link_libraries(DERAPA_Tests PUBLIC mimalloc)
	target_link_libraries(DERAPA_Tests PUBLIC mimalloc-static)
	target_link_libraries(DERAPA_Tests PUBLIC /usr/local/lib/mimalloc-2.0/mimalloc.o)
	set(ENV{LD_PRELOAD} /usr/local/lib/libmimalloc.so.2.0)
	set(ENV{MIMALLOC_PAGE_RESET} 0)
	set(ENV{HUGETLB_MORECORE} thp)
	set(ENV{MIMALLOC_LARGE_OS_PAGES} 1)
	set(ENV{MIMALLOC_RESERVE_HUGE_OS_PAGES} 16)
	set(ENV{MIMALLOC_SHOW_STATS} 1)
	set(ENV{MIMALLOC_VERBOSE} 1)
else()
	ExternalProject_Add(mimalloc
		URL https://github.com/microsoft/mimalloc/archive/refs/heads/master.zip
		CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${DERAPA_BINARY_DIR}/deps -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
	)
	add_dependencies(DERAPA_Tests mimalloc)
	target_include_directories(DERAPA_Tests SYSTEM PUBLIC ${DERAPA_BINARY_DIR}/deps/include/mimalloc-2.0/)
	target_link_libraries(DERAPA_Tests PUBLIC ${DERAPA_BINARY_DIR}/deps/lib/mimalloc-2.0/mimalloc.o)
	set(ENV{LD_PRELOAD} ${DERAPA_BINARY_DIR}/deps/lib/libmimalloc.so.2.0)
	set(ENV{MIMALLOC_PAGE_RESET} 0)
	set(ENV{HUGETLB_MORECORE} thp)
	set(ENV{MIMALLOC_LARGE_OS_PAGES} 1)
	set(ENV{MIMALLOC_RESERVE_HUGE_OS_PAGES} 16)
	set(ENV{MIMALLOC_SHOW_STATS} 1)
	set(ENV{MIMALLOC_VERBOSE} 1)
endif()

if(Boost_FOUND)
	target_link_libraries(DERAPA PUBLIC Boost::program_options)
endif()

set_property(TARGET DERAPA_Tests PROPERTY CXX_STANDARD 17)

# target_compile_options(DERAPA_Tests PUBLIC -O3 -march=native)

add_dependencies(DERAPA_Tests catch2)
target_include_directories(DERAPA_Tests SYSTEM PUBLIC ${DERAPA_BINARY_DIR}/deps/include)
target_include_directories(DERAPA_Tests SYSTEM PUBLIC ${DERAPA_BINARY_DIR}/deps/include/mimalloc-2.0/)

if(APPLE)
	target_link_libraries(DERAPA_Tests PUBLIC "-framework CoreFoundation")
endif(APPLE)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
	target_link_libraries(DERAPA_Tests PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(DERAPA_Tests PUBLIC ittnotify)
# target_link_libraries(DERAPA_Tests PUBLIC -ljemalloc)