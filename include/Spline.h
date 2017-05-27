#ifndef FLIGHT_SPLINE_H
#define FLIGHT_SPLINE_H

#include <vector>
#include <Eigen/Dense>

class Spline {
    std::vector<Eigen::MatrixXd> points;
    std::vector<Eigen::MatrixXd> coefficients;
    unsigned long int num_points;
    unsigned long int num_dimensions;
    unsigned long int num_segments;
public:
    Spline(std::vector<std::vector<double>> points);
    std::vector<double> position_at_time(double t);
    std::vector<double> derivative_at_time(double t);
    std::vector<double> double_derivative_at_time(double t);
    std::vector<std::vector<double>> get_points();
    unsigned long int get_num_dimensions();
    unsigned long int get_num_segments();
    std::vector<std::vector<double>> get_coefficients();
    std::vector<double> closest_point(std::vector<double> point, double time = 0.0f);
    double closest_time(std::vector<double> point, double time = 0.0f);

private:
    void _initialise_spline();
    Eigen::MatrixXd _position_at_time(double t);
};


#endif //FLIGHT_SPLINE_H
