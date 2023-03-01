#ifndef GDEL2D_EDIT_TRIANGULATIONHANDLER_H
#define GDEL2D_EDIT_TRIANGULATIONHANDLER_H

#include "DelaunayChecker.h"
#include "GPU/GpuDelaunay.h"
#include "InputCreator.h"
#include "PerfTimer.h"
#include <iomanip>
#include <bits/stdc++.h>


struct line {
    Point2 p1, p2;
};

inline bool onLine(line l1, Point2 p)
{
    // Check whether p is on the line or not
    if (p._p[0] <= std::max(l1.p1._p[0], l1.p2._p[0])
        && p._p[0] <= std::min(l1.p1._p[0], l1.p2._p[0])
        && (p._p[1] <= std::max(l1.p1._p[1], l1.p2._p[1])
            && p._p[1] <= std::min(l1.p1._p[1], l1.p2._p[1])))
        return true;

    return false;
}

inline int direction(Point2 a, Point2 b, Point2 c)
{
    auto val = (b._p[1] - a._p[1]) * (c._p[0] - b._p[0])
              - (b._p[0] - a._p[0]) * (c._p[1] - b._p[1]);

    if (val == 0)

        // Colinear
        return 0;

    else if (val < 0)

        // Anti-clockwise direction
        return 2;

    // Clockwise direction
    return 1;
}

inline bool isIntersect(line l1, line l2)
{
    // Four direction for two lines and points of other line
    int dir1 = direction(l1.p1, l1.p2, l2.p1);
    int dir2 = direction(l1.p1, l1.p2, l2.p2);
    int dir3 = direction(l2.p1, l2.p2, l1.p1);
    int dir4 = direction(l2.p1, l2.p2, l1.p2);

    // When intersecting
    if (dir1 != dir2 && dir3 != dir4)
        return true;

    // When p2 of line2 are on the line1
    if (dir1 == 0 && onLine(l1, l2.p1))
        return true;

    // When p1 of line2 are on the line1
    if (dir2 == 0 && onLine(l1, l2.p2))
        return true;

    // When p2 of line1 are on the line2
    if (dir3 == 0 && onLine(l2, l1.p1))
        return true;

    // When p1 of line1 are on the line2
    if (dir4 == 0 && onLine(l2, l1.p2))
        return true;

    return false;
}

class TriangulationHandler
{
  private:
    // Main
    int  _runNum  = 1;
    bool _doCheck = false;

    InputCreatorPara PtCreatorPara;

    bool        _outputResult = false;
    std::string _outCheckFilename;
    std::string _outMeshFilename;
    double      InitX = 0;
    double      InitY = 0;
    double      InitZ = 0;

    // In-Out Data
    GDel2DInput  _input;
    GDel2DOutput _output;

#ifndef DISABLE_PCL_INPUT
    pcl::PointCloud<POINT_TYPE> InputPointCloud;
#endif

    // Statistics
    Statistics statSum;

//    HashTable<uint32_t, Tri> PtToTriTable;

    TriangulationHandler() = default;
    void reset();
    void saveResultsToFile();
    bool checkInside(Tri &t, Point2 p) const;
    Eigen::Vector3d getGroundNorm() const;

    static double getupwards(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3);

    static Eigen::Vector3d getTriNormal(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3);

    Eigen::Vector3d getTriNormal(const Tri &t) const;

    static bool hasValidEdge(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3);

  public:
    explicit TriangulationHandler(const char *InputYAMLFile);
    void run();
};

#endif //GDEL2D_EDIT_TRIANGULATIONHANDLER_H