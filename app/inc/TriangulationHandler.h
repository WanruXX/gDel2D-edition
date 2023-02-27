#ifndef GDEL2D_EDIT_TRIANGULATIONHANDLER_H
#define GDEL2D_EDIT_TRIANGULATIONHANDLER_H

#include "DelaunayChecker.h"
#include "GPU/GpuDelaunay.h"
#include "InputCreator.h"
#include "PerfTimer.h"
#include <iomanip>

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

    TriangulationHandler() = default;
    void reset();
    void saveResultsToFile() const;

    static double getTriNormal(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3);

    static bool hasValidEdge(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3);

  public:
    explicit TriangulationHandler(const char *InputYAMLFile);
    void run();
};

#endif //GDEL2D_EDIT_TRIANGULATIONHANDLER_H