# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rohit/Deep_Learning/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rohit/Deep_Learning/project

# Include any dependencies generated for this target.
include CMakeFiles/compose_panaroma.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/compose_panaroma.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/compose_panaroma.dir/flags.make

CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o: CMakeFiles/compose_panaroma.dir/flags.make
CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o: compose_panaroma.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rohit/Deep_Learning/project/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o -c /home/rohit/Deep_Learning/project/compose_panaroma.cpp

CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rohit/Deep_Learning/project/compose_panaroma.cpp > CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.i

CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rohit/Deep_Learning/project/compose_panaroma.cpp -o CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.s

CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o.requires:
.PHONY : CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o.requires

CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o.provides: CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o.requires
	$(MAKE) -f CMakeFiles/compose_panaroma.dir/build.make CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o.provides.build
.PHONY : CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o.provides

CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o.provides.build: CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o

# Object files for target compose_panaroma
compose_panaroma_OBJECTS = \
"CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o"

# External object files for target compose_panaroma
compose_panaroma_EXTERNAL_OBJECTS =

compose_panaroma: CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o
compose_panaroma: CMakeFiles/compose_panaroma.dir/build.make
compose_panaroma: /usr/local/lib/libopencv_videostab.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_video.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_ts.a
compose_panaroma: /usr/local/lib/libopencv_superres.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_stitching.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_photo.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_ocl.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_objdetect.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_nonfree.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_ml.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_legacy.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_imgproc.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_highgui.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_gpu.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_flann.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_features2d.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_core.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_contrib.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_calib3d.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_videostab.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_video.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_ts.a
compose_panaroma: /usr/local/lib/libopencv_superres.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_stitching.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_photo.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_ocl.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_objdetect.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_nonfree.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_ml.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_legacy.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_imgproc.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_highgui.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_gpu.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_flann.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_features2d.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_core.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_contrib.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_calib3d.so.2.4.13
compose_panaroma: /usr/lib/x86_64-linux-gnu/libGLU.so
compose_panaroma: /usr/lib/x86_64-linux-gnu/libGL.so
compose_panaroma: /usr/lib/x86_64-linux-gnu/libSM.so
compose_panaroma: /usr/lib/x86_64-linux-gnu/libICE.so
compose_panaroma: /usr/lib/x86_64-linux-gnu/libX11.so
compose_panaroma: /usr/lib/x86_64-linux-gnu/libXext.so
compose_panaroma: /usr/local/lib/libopencv_nonfree.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_ocl.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_gpu.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_photo.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_objdetect.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_legacy.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_video.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_ml.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_calib3d.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_features2d.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_highgui.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_imgproc.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_flann.so.2.4.13
compose_panaroma: /usr/local/lib/libopencv_core.so.2.4.13
compose_panaroma: CMakeFiles/compose_panaroma.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable compose_panaroma"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compose_panaroma.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/compose_panaroma.dir/build: compose_panaroma
.PHONY : CMakeFiles/compose_panaroma.dir/build

CMakeFiles/compose_panaroma.dir/requires: CMakeFiles/compose_panaroma.dir/compose_panaroma.cpp.o.requires
.PHONY : CMakeFiles/compose_panaroma.dir/requires

CMakeFiles/compose_panaroma.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/compose_panaroma.dir/cmake_clean.cmake
.PHONY : CMakeFiles/compose_panaroma.dir/clean

CMakeFiles/compose_panaroma.dir/depend:
	cd /home/rohit/Deep_Learning/project && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rohit/Deep_Learning/project /home/rohit/Deep_Learning/project /home/rohit/Deep_Learning/project /home/rohit/Deep_Learning/project /home/rohit/Deep_Learning/project/CMakeFiles/compose_panaroma.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/compose_panaroma.dir/depend

