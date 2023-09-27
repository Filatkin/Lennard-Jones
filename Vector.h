#ifndef LENNARD_JONES_Vector_H
#define LENNARD_JONES_Vector_H


#include <iostream>

class Vector {
public:
    double x, y, z;

    explicit Vector(double x, double y, double z);

    Vector operator*(const double &c) const;

    Vector operator+(const Vector &v) const;

    Vector operator+(const double &c) const;

    Vector operator-(const Vector &v) const;

    double operator*(const Vector &v) const;
};

Vector operator*(double c, const Vector &v);

std::ostream &operator<<(std::ostream &os, const Vector &v);

#endif //LENNARD_JONES_Vector_H
