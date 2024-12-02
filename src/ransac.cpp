#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

struct Point {
    double x, y;
};

struct Line {
    double slope, intercept;
};

double calculateDistance(const Point& p, const Line& line) {
    // Calculate perpendicular distance from a point to a line.
    return std::abs(line.slope * p.x - p.y + line.intercept) / std::sqrt(line.slope * line.slope + 1);
}

Line fitLine(const Point& p1, const Point& p2) {
    // Fit a line through two points
    double slope = (p2.y - p1.y) / (p2.x - p1.x);
    double intercept = p1.y - slope * p1.x;
    return {slope, intercept};
}

Line ransac(const std::vector<Point>& points, int maxIterations, double distanceThreshold, int& bestInlierCount) {
    Line bestLine = {0, 0};
    bestInlierCount = 0;
    std::srand(static_cast<unsigned>(std::time(0)));

    for (int i = 0; i < maxIterations; ++i) {
        // Randomly select two points
        int idx1 = std::rand() % points.size();
        int idx2 = std::rand() % points.size();
        while (idx1 == idx2) {
            idx2 = std::rand() % points.size();
        }

        Line line = fitLine(points[idx1], points[idx2]);
        int inlierCount = 0;

        for (const auto& p : points) {
            if (calculateDistance(p, line) < distanceThreshold) {
                ++inlierCount;
            }
        }

        if (inlierCount > bestInlierCount) {
            bestLine = line;
            bestInlierCount = inlierCount;
        }
    }

    return bestLine;
}

int main() {
    // Example points (you can modify or extend this)
    std::vector<Point> points = {
        {1, 2}, {2, 4.1}, {3, 5.9}, {4, 8}, {5, 10}, {100, 100}, {101, 101}
    };

    int maxIterations = 100;
    double distanceThreshold = 1.0;
    int bestInlierCount = 0;

    Line bestLine = ransac(points, maxIterations, distanceThreshold, bestInlierCount);

    std::cout << "Best line: y = " << bestLine.slope << "x + " << bestLine.intercept << "\n";
    std::cout << "Number of inliers: " << bestInlierCount << "\n";

    return 0;
}
