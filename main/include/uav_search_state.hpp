#ifndef UAV_SEARCH_STATE_HPP
#define UAV_SEARCH_STATE_HPP

#include <iostream>
#include <cmath>
#include <nlohmann/json.hpp>

struct UAVSearchState {
    double prob;
    int    uav;
    UAVSearchState() : prob(0.0), uav(0) {}
};

inline bool operator!=(const UAVSearchState& x, const UAVSearchState& y) {
    return x.uav != y.uav || std::abs(x.prob - y.prob) > 1e-9;
}

inline std::ostream& operator<<(std::ostream& os, const UAVSearchState& s) {
    os << s.prob << "," << s.uav;    // comma + space — required by Cadmium viewer
    return os;
}

[[maybe_unused]] inline void from_json(const nlohmann::json& j, UAVSearchState& s) {
    if (j.contains("prob")) j.at("prob").get_to(s.prob);
    if (j.contains("uav"))  j.at("uav").get_to(s.uav);
}

#endif
