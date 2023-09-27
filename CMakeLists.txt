cmake_minimum_required(VERSION 3.26)
project(Lennard_Jones)

set(CMAKE_CXX_STANDARD 23)

add_executable(Lennard_Jones main.cpp
        Vector.cpp
        Vector.h)
