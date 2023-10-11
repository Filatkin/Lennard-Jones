#include "Vector.h"

Vector::Vector(double x_, double y_, double z_) {
    x = x_, y = y_, z = z_;
}

Vector::Vector() = default;

Vector Vector::operator*(const double &c) const {
    return Vector(x * c, y * c, z * c);
}

Vector Vector::operator/(const double &c) const {
    return Vector(x / c, y / c, z / c);
}

Vector Vector::operator+(const Vector &v) const {
    return Vector(x + v.x, y + v.y, z + v.z);;
}

Vector Vector::operator+(const double &c) const {
    return Vector(x + c, y + c, z + c);;
}

Vector Vector::operator-(const Vector &v) const {
    return Vector(x - v.x, y - v.y, z - v.z);
}

double Vector::operator*(const Vector &v) const {
    return x * v.x + y * v.y + z * v.z;
}

Vector operator*(const double c, const Vector &v) {
    return Vector(v.x * c, v.y * c, v.z * c);
}

std::ostream &operator<<(std::ostream &os, const Vector &v) {
    os << "(" << v.x << ";" << v.y << ";" << v.z << ")";
    return os;
}