#include "Spline.h"

#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/StdVector>


/**
 * Helper method to convert Eigen::MatrixXds to std::vectors
 */
std::vector<double> matrix_to_vector(Eigen::MatrixXd matrix) {
    std::vector<double> r;
    for (auto i=0; i < matrix.size(); i++) {
        r.push_back(matrix(i));
    }
    return r;
}

/**
 * Helper method to convert std::vectors to Eigen::MatrixXds
 */
Eigen::MatrixXd vector_to_matrix(std::vector<double> vector) {
    return Eigen::Map<Eigen::MatrixXd>(&vector[0], 1, vector.size());
}

/**
 * Constructor, takes a nested vector of points that will define the line.
 */
Spline::Spline(std::vector<std::vector<double>> points) {
    std::vector<Eigen::MatrixXd> out_points;
    for (auto i=0; i < points.size(); i++) {
        Eigen::MatrixXd copy = Eigen::Map<Eigen::MatrixXd>(&points[i][0], 1, points[i].size());
        out_points.push_back(copy);
    }
    this->points = out_points;
    this->num_points = points.size();
    this->num_dimensions = points[0].size();
    this->num_segments = num_points - 1;
    for (auto i = 0; i < this->num_segments; i++) {
        Eigen::MatrixXd a(4, this->num_dimensions);
        this->coefficients.push_back(a);
    }
    _initialise_spline(); // coefficients are stored [d, c, b, a]
}

/**
 * This method solves for the cubic coefficients for the spline for the
 * points given in the constructor
 */
void Spline::_initialise_spline() {
    std::vector<Eigen::MatrixXd> a;
    for (auto i = 0; i < this->num_points; i++) {
        Eigen::MatrixXd aa = Eigen::MatrixXd::Zero(1, this->num_dimensions);
        a.push_back(aa);
    }
    for (auto i = 1; i < this->num_segments; i++) {
        Eigen::MatrixXd y = - 2 * this->points[i];
        Eigen::MatrixXd q = this->points[i+1] + y + this->points[i - 1];
        a[i] = 3*q;
    }
    Eigen::MatrixXd l = Eigen::MatrixXd::Zero(1, this->num_points);
    Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(1, this->num_points);
    std::vector<Eigen::MatrixXd> z;
    for (auto i = 0; i < this->num_points; i++) {
        Eigen::MatrixXd f= Eigen::MatrixXd::Zero(1, this->num_dimensions);
        z.push_back(f);
    }

    l(0), l(this->num_segments) = 1;

    for (auto i = 1; i < this->num_segments; i++) {
        l(i) = 4 - mu(i - 1);
        mu(i) = 1/l(i);
        Eigen::MatrixXd d = a[i] - z[i - 1];
        z[i] = d/l(i);
    }

    for (auto i=0; i < this->num_segments; i++) {
        this->coefficients[i].row(0) = this->points[i];
    }

    // note that we are decrementing on an unsigned int (hence the decrement prior to comparison to 0)
    // also note that this means that the first value of i is num_segments - 1 (which may not be immediately obvious)
    for (auto i = this->num_segments; i --> 0 ;) {

        if (i < this->num_segments - 1) {
            this->coefficients[i].row(2) = z[i] - (mu(i) * this->coefficients[i + 1].row(2));
            this->coefficients[i].row(3) = (1.0 / 3.0) * (this->coefficients[i + 1].row(2) - this->coefficients[i].row(2));
            this->coefficients[i].row(1) = (this->points[i + 1] - this->points[i]) - (this->coefficients[i].row(2) + this->coefficients[i].row(3));
        } else {
            this->coefficients[i].row(2) = z[i];
            this->coefficients[i].row(3) = -this->coefficients[i].row(2)/3;
            this->coefficients[i].row(1) = (this->points[i + 1] - this->points[i]) - (this->coefficients[i].row(2) + this->coefficients[i].row(3));
        }
    }
}

/**
 * Given a time t, this method will return a std::vector containing the
 * point along the spline that occurs at time t
 *
 * @param t the time to check
 * @return std::vector containing the point
 */
std::vector<double> Spline::position_at_time(double t) {
    return matrix_to_vector(this->_position_at_time(t));
}

/**
 * Private method that returns an Eigen::MatrixXd of the point at time t
 *
 * @param t the time to check
 * @return Eigen::Matrix containing the point
 */
Eigen::MatrixXd Spline::_position_at_time(double t) {
    if (t > this->num_points) {
        t = this->num_points;
    }

    auto segment = floor(t);
    auto x = t - segment;

    segment = fmod(segment, this->num_segments);

    Eigen::MatrixXd position = Eigen::MatrixXd::Zero(1, this->num_dimensions);
    for (auto i = 0; i < 4; i++) {
        position += this->coefficients[segment].row(i) * std::pow(x, i);
    }
    return position;
}

/**
 * Returns the derivative at the given time
 *
 * @param t the time to check
 * @return std::vector containing the derivative at time t
 */
std::vector<double> Spline::derivative_at_time(double t) {
    if (t > this->num_points) {
        t = this->num_points;
    }

    auto segment = floor(t);
    auto x = t - segment;

    segment = fmod(segment, this->num_segments);

    Eigen::MatrixXd derivative = Eigen::MatrixXd::Zero(1, this->num_dimensions);
    for (auto i = 1; i < 4; i++) {
        derivative += i * this->coefficients[segment].row(i) * std::pow(x, i-1);
    }

    return matrix_to_vector(derivative);
}

/**
 * Returns the second derivative at the given time
 *
 * @param t the time to check
 * @return std::vector containing the second derivative at time t
 */
std::vector<double> Spline::double_derivative_at_time(double t) {
    if (t > this->num_points) {
        t = this->num_points;
    }

    auto segment = floor(t);
    auto x = t - segment;

    segment = fmod(segment, this->num_segments);

    Eigen::MatrixXd double_derivative = Eigen::MatrixXd::Zero(1, this->num_dimensions);
    for (auto i = 2; i < 4; i++) {
        double_derivative += i * (i - 1) * this->coefficients[segment].row(i) * std::pow(x, i-2);
    }

    return matrix_to_vector(double_derivative);
}

/**
 * Returns the points that define this spline
 * The points are in order from t = 0 and up :)
 * @return nested vector of the points
 */
std::vector<std::vector<double>> Spline::get_points() {
    std::vector<std::vector<double>> return_points;
    for (auto &point : this->points) {
        return_points.push_back(matrix_to_vector(point));
    }
    return return_points;
}

/**
 * Returns the coefficients that define this spline
 *
 * for the cubic function denoted as:
 *
 * ax^3 + bx^2 + cx + d
 *
 * the coefficents are stored [d, c, b, a] in order of the
 * segments from t=0 and up
 *
 * @return nested vector of the coefficients
 */
std::vector<std::vector<double>> Spline::get_coefficients() {
    std::vector<std::vector<double>> return_coefficients;
    for (auto &coefficient : this->coefficients) {
        return_coefficients.push_back(matrix_to_vector(coefficient));
    }
    return return_coefficients;
}

/**
 * Returns the closest point along the spline to a given point that
 * may or may not belong on the spline
 *
 * @param point the point to check against
 * @param time a time to guide the selection of the point // todo make this optional
 * @return the closest point along the spline in the form of a std::vector :)
 */
std::vector<double> Spline::closest_point(std::vector<double> point, double time) {
    return this->position_at_time(this->closest_time(point, time));
}

/**
 * Returns the closest time along the spline to a given point that
 * may or may not belong on the spline
 *
 * @param point the point to check against
 * @param time a time to guide the selection of the point // todo make this optional
 * @return the closest point along the spline in the form of a std::vector :)
 */
double Spline::closest_time(std::vector<double> point, double time) {
    // TODO fix this method, it's awful and hacky
    Eigen::MatrixXd p = vector_to_matrix(point);
    double min_dist = 9999999;
    Eigen::MatrixXd closest_point;
    for (auto i=0; i < 100; i++) {
        double ii = i /10000;
        Eigen::MatrixXd sat = this->_position_at_time(time + ii);
        Eigen::MatrixXd d = p - sat;
        double dist = d.norm();
        if (dist < min_dist) {
            min_dist = dist;
            closest_point = sat;
            time = time + ii;
        }

    }
    return time;
}

/**
 * un comment this code if you wish to run the spline executable, otherwise,
 * completely unneccessary
 */
//int main() {
//    std::vector<std::vector<double>> p = {{-27.393183401605732, 153.12379360198975},
//                                          {-27.367498827657315, 153.138427734375},
//                                          {-27.323584984266567, 153.14701080322266},
//                                          {-27.305587481685592, 153.07096481323242},
//                                          {-27.322364906869076, 152.9241943359375},
//                                          {-27.486953781553893, 152.98255920410156}};
//    Spline s(p);
//    std::vector<std::vector<double>> points = s.get_points();
//    for (int i=0; i < points.size(); i++) {
//        printf("%f, %f\n", points[i][0], points[i][1]);
//    }
//    double t = 0.5;
//    std::vector<double> pos = s.position_at_time(t);
//    printf("position at time %f, lat: %f, lng: %f \n", t, pos[0], pos[1]);
//    double ct = s.closest_time(pos, t);
//    printf("closest time %f\n", ct);
////    printf("closest point lat: %f, lng: %f \n", cp[0], cp[1]);
//    return 0;
//}