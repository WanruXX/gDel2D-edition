#include "../inc/TriangulationHandler.h"
#include "../inc/json.h"
#include <Eigen/Dense>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

TriangulationHandler::TriangulationHandler(const char *InputYAMLFile)
{
    YAML::Node Config = YAML::LoadFile(InputYAMLFile);

    InitX = Config["MapOriginInUTM"]["X"].as<double>();
    InitY = Config["MapOriginInUTM"]["Y"].as<double>();
    InitZ = Config["MapOriginInUTM"]["Z"].as<double>();

    _runNum  = Config["RunNum"].as<int>();
    _doCheck = Config["DoCheck"].as<bool>();

    PtCreatorPara._inFile = Config["InputFromFile"].as<bool>();
    if (PtCreatorPara._inFile)
    {
        PtCreatorPara._inFilename           = Config["InputPointCloudFile"].as<std::string>();
        PtCreatorPara._inConstraintFilename = Config["InputConstraintFile"].as<std::string>();
        if (access(PtCreatorPara._inFilename.c_str(), F_OK) == -1)
        {
            std::cerr << "Input point cloud file " << PtCreatorPara._inFilename << " doesn't exist, generate points..."
                      << std::endl;
            PtCreatorPara._inFile = false;
        }
        else
        {
            if (access(PtCreatorPara._inConstraintFilename.c_str(), F_OK) == -1)
            {
                std::cerr << "Input constraints file " << PtCreatorPara._inConstraintFilename
                          << " doesn't exist, not using constraints..." << std::endl;
            }
            else
            {
                PtCreatorPara._constraintNum = 0;
            }
        }
    }
    else
    {
        PtCreatorPara._pointNum      = Config["PointNum"].as<int>();
        std::string DistributionType = Config["DistributionType"].as<std::string>();
        if (DistributionType == "Gaussian")
        {
            PtCreatorPara._dist = GaussianDistribution;
        }
        else if (DistributionType == "Disk")
        {
            PtCreatorPara._dist = DiskDistribution;
        }
        else if (DistributionType == "ThinCircle")
        {
            PtCreatorPara._dist = ThinCircleDistribution;
        }
        else if (DistributionType == "Circle")
        {
            PtCreatorPara._dist = CircleDistribution;
        }
        else if (DistributionType == "Grid")
        {
            PtCreatorPara._dist = GridDistribution;
        }
        else if (DistributionType == "Ellipse")
        {
            PtCreatorPara._dist = EllipseDistribution;
        }
        else if (DistributionType == "TwoLines")
        {
            PtCreatorPara._dist = TwoLineDistribution;
        }
        PtCreatorPara._constraintNum = Config["ConstraintNum"].as<int>();
        PtCreatorPara._seed          = Config["Seed"].as<int>();
        PtCreatorPara._saveToFile    = Config["SaveToFile"].as<bool>();
        PtCreatorPara._savePath      = Config["SavePath"].as<std::string>();
    }

    _input.insAll              = Config["InsertAll"].as<bool>();
    _input.noSort              = Config["NoSortPoint"].as<bool>();
    _input.noReorder           = Config["NoReorder"].as<bool>();
    std::string InputProfLevel = Config["ProfLevel"].as<std::string>();
    if (InputProfLevel == "Detail")
    {
        _input.profLevel = ProfDetail;
    }
    else if (InputProfLevel == "Diag")
    {
        _input.profLevel = ProfDiag;
    }
    else if (InputProfLevel == "Debug")
    {
        _input.profLevel = ProfDebug;
    }

    _outputResult     = Config["OutputResult"].as<bool>();
    _outCheckFilename = Config["OutputCheckResult"].as<std::string>();
    _outMeshFilename  = Config["OutputMeshPath"].as<std::string>();
}

void TriangulationHandler::reset()
{
    Point2HVec().swap(_input.InputPointVec);
    SegmentHVec().swap(_input.InputConstraintVec);
    TriHVec().swap(_output.triVec);
    TriOppHVec().swap(_output.triOppVec);
    cudaDeviceReset();
}

void TriangulationHandler::run()
{
    // Pick the best CUDA device
    const int deviceIdx = cutGetMaxGflopsDeviceId();
    CudaSafeCall(cudaSetDevice(deviceIdx));
    CudaSafeCall(cudaDeviceReset());

    GpuDel gpuDel;
    for (int i = 0; i < _runNum; ++i)
    {
        reset();
        // 1. Create points
        auto         timer = (double)clock();
        InputCreator creator;
#ifndef DISABLE_PCL_INPUT
        creator.createPoints(PtCreatorPara, InputPointCloud, _input.InputPointVec, _input.InputConstraintVec);
#else
        creator.createPoints(PtCreatorPara, _input.InputPointVec, _input.InputConstraintVec);
#endif
        std::cout << "Point reading time: " << ((double)clock() - timer) / CLOCKS_PER_SEC << std::endl;
        // 2. Compute Delaunay triangulation
        timer = (double)clock();
        gpuDel.compute(_input, &_output);
        std::cout << "Delaunay computing time: " << ((double)clock() - timer) / CLOCKS_PER_SEC << std::endl;
        const Statistics &stats = _output.stats;
        statSum.accumulate(stats);
        std::cout << "\nTIME: " << stats.totalTime << "(" << stats.initTime << ", " << stats.splitTime << ", "
                  << stats.flipTime << ", " << stats.relocateTime << ", " << stats.sortTime << ", "
                  << stats.constraintTime << ", " << stats.outTime << ")" << std::endl;

        if (_doCheck)
        {
            DelaunayChecker checker(_input, _output);
            std::cout << "\n*** Check ***\n";
            checker.checkEuler();
            checker.checkOrientation();
            checker.checkAdjacency();
            checker.checkConstraints();
            checker.checkDelaunay();
        }
        ++PtCreatorPara._seed;
    }
    statSum.average(_runNum);

    if (_outputResult)
    {
        saveResultsToFile();
    }

    std::cout << std::endl;
    std::cout << "---- SUMMARY ----" << std::endl;
    std::cout << std::endl;
    std::cout << "PointNum       " << _input.InputPointVec.size() << std::endl;
    std::cout << "Sort           " << (_input.noSort ? "no" : "yes") << std::endl;
    std::cout << "Reorder        " << (_input.noReorder ? "no" : "yes") << std::endl;
    std::cout << "Insert mode    " << (_input.insAll ? "InsAll" : "InsFlip") << std::endl;
    std::cout << std::endl;
    std::cout << std::fixed << std::right << std::setprecision(2);
    std::cout << "TotalTime (ms) " << std::setw(10) << statSum.totalTime << std::endl;
    std::cout << "InitTime       " << std::setw(10) << statSum.initTime << std::endl;
    std::cout << "SplitTime      " << std::setw(10) << statSum.splitTime << std::endl;
    std::cout << "FlipTime       " << std::setw(10) << statSum.flipTime << std::endl;
    std::cout << "RelocateTime   " << std::setw(10) << statSum.relocateTime << std::endl;
    std::cout << "SortTime       " << std::setw(10) << statSum.sortTime << std::endl;
    std::cout << "ConstraintTime " << std::setw(10) << statSum.constraintTime << std::endl;
    std::cout << "OutTime        " << std::setw(10) << statSum.outTime << std::endl;
    std::cout << std::endl;
}

void TriangulationHandler::saveResultsToFile() const
{
    if (_outputResult)
    {
        std::ofstream CheckOutput(_outCheckFilename, std::ofstream::app);
        if (CheckOutput.is_open())
        {
            CheckOutput << "GridWidth," << GridSize << ",";
            CheckOutput << "PointNum," << _input.InputPointVec.size() << ",";
            CheckOutput << "Runs," << _runNum << ",";
            CheckOutput << "Input,"
                        << (PtCreatorPara._inFile ? PtCreatorPara._inFilename : DistStr[PtCreatorPara._dist]) << ",";
            CheckOutput << (_input.noSort ? "--" : "sort") << ",";
            CheckOutput << (_input.noReorder ? "--" : "reorder") << ",";
            CheckOutput << (_input.insAll ? "InsAll" : "--") << ",";

            CheckOutput << "TotalTime," << statSum.totalTime / 1000.0 << ",";
            CheckOutput << "InitTime," << statSum.initTime / 1000.0 << ",";
            CheckOutput << "SplitTime," << statSum.splitTime / 1000.0 << ",";
            CheckOutput << "FlipTime," << statSum.flipTime / 1000.0 << ",";
            CheckOutput << "RelocateTime," << statSum.relocateTime / 1000.0 << ",";
            CheckOutput << "SortTime," << statSum.sortTime / 1000.0 << ",";
            CheckOutput << "ConstraintTime," << statSum.constraintTime / 1000.0 << ",";
            CheckOutput << "OutTime," << statSum.outTime / 1000.0;
            CheckOutput << std::endl;

            CheckOutput.close();
        }
        else
        {
            std::cerr << _outCheckFilename << " is not a valid path!" << std::endl;
        }

        std::ofstream MeshOutput(_outMeshFilename);
        std::string   MeshData;
        if (MeshOutput.is_open())
        {

            MeshData =
                R"({"crs":{"properties":{"name":"urn:ogc:def:crs:EPSG::5556"},"type":"name"},"name":"left_4_edge_polygon","type":"FeatureCollection", "features":[)";
            MeshData += "\n";
            //            nlohmann::json JsonFile;
            //            JsonFile["type"]                      = "FeatureCollection";
            //            JsonFile["name"]                      = "left_4_edge_polygon";
            //            JsonFile["crs"]["type"]               = "name";
            //            JsonFile["crs"]["properties"]["name"] = "urn:ogc:def:crs:EPSG::5556";
            //            JsonFile["features"]                  = nlohmann::json::array();
            int TriID = 0;
            for (auto &tri : _output.triVec)
            {
                if (hasValidEdge(InputPointCloud[tri._v[0]], InputPointCloud[tri._v[1]], InputPointCloud[tri._v[2]]))
                {
                    MeshData +=
                        R"({ "type": "Feature", "properties": { "TriID": )" + std::to_string(TriID) +
                        " , \"Upward\": " +
                        std::to_string(getTriNormal(
                            InputPointCloud[tri._v[0]], InputPointCloud[tri._v[1]], InputPointCloud[tri._v[2]])) +
                        R"( }, "geometry": { "type": "Polygon", "coordinates": [[[)" +
                        std::to_string(static_cast<double>(InputPointCloud[tri._v[0]].x) + InitX) + ", " +
                        std::to_string(static_cast<double>(InputPointCloud[tri._v[0]].y) + InitY) + "], [" +
                        std::to_string(static_cast<double>(InputPointCloud[tri._v[1]].x) + InitX) + ", " +
                        std::to_string(static_cast<double>(InputPointCloud[tri._v[1]].y) + InitY) + "], [" +
                        std::to_string(static_cast<double>(InputPointCloud[tri._v[2]].x) + InitX) + ", " +
                        std::to_string(static_cast<double>(InputPointCloud[tri._v[2]].y) + InitY) + "]]] } },\n";
//                    nlohmann::json Coor = nlohmann::json::array();
#ifndef DISABLE_PCL_INPUT
//                    Coor.push_back({static_cast<double>(InputPointCloud[tri._v[0]].x) + InitX,
//                                    static_cast<double>(InputPointCloud[tri._v[0]].y) + InitY,
//                                    static_cast<double>(InputPointCloud[tri._v[0]].z) + InitZ});
//                    Coor.push_back({static_cast<double>(InputPointCloud[tri._v[1]].x) + InitX,
//                                    static_cast<double>(InputPointCloud[tri._v[1]].y) + InitY,
//                                    static_cast<double>(InputPointCloud[tri._v[1]].z) + InitZ});
//                    Coor.push_back({static_cast<double>(InputPointCloud[tri._v[2]].x) + InitX,
//                                    static_cast<double>(InputPointCloud[tri._v[2]].y) + InitY,
//                                    static_cast<double>(InputPointCloud[tri._v[2]].z) + InitZ});
#else
                    Coor.push_back({static_cast<double>(_input.InputPointVec[tri._v[0]]._p[0]) + InitX,
                                    static_cast<double>(_input.InputPointVec[tri._v[0]]._p[1]) + InitY});
                    Coor.push_back({static_cast<double>(_input.InputPointVec[tri._v[1]]._p[0]) + InitX,
                                    static_cast<double>(_input.InputPointVec[tri._v[1]]._p[1]) + InitY});
                    Coor.push_back({static_cast<double>(_input.InputPointVec[tri._v[2]]._p[0]) + InitX,
                                    static_cast<double>(_input.InputPointVec[tri._v[2]]._p[1]) + InitY});
#endif
                    //                    nlohmann::json CoorWrapper = nlohmann::json::array();
                    //                    CoorWrapper.push_back(Coor);
                    //                    nlohmann::json TriangleObject;
                    //                    TriangleObject["type"]       = "Feature";
                    //                    TriangleObject["properties"] = {
                    //                        {"TriID", TriID},
                    //                        {"Upward",
                    //                         std::to_string(getTriNormal(
                    //                             InputPointCloud[tri._v[0]], InputPointCloud[tri._v[1]], InputPointCloud[tri._v[2]]))}};
                    //                    TriangleObject["geometry"] = {{"type", "Polygon"}, {"coordinates", CoorWrapper}};
                    //                    JsonFile["features"].push_back(TriangleObject);
                    ++TriID;
                }
            }
                //                MeshOutput << JsonFile << std::endl;
                MeshData.pop_back();
                MeshData.pop_back();
                MeshOutput << MeshData;
                MeshOutput << "]\n}" << std::endl;
                MeshOutput.close();

            //            std::ofstream OutputStream(OutputFile);
            //            nlohmann::json JsonFile;
            //            JsonFile["type"] = "FeatureCollection";
            //            JsonFile["name"] = "left_4_edge";
            //            JsonFile["crs"]["type"] = "name";
            //            JsonFile["crs"]["properties"]["name"] = "urn:ogc:def:crs:EPSG::5556";
            //            JsonFile["features"] = nlohmann::json::array();
            //            int TriID = 0;
            //            for (auto &seg: segSet) {
            //                nlohmann::json Coor = nlohmann::json::array();
            //                Coor.push_back({static_cast<double>(pointVec[seg._v[0]]._p[0]) + InitX,
            //                                static_cast<double>(pointVec[seg._v[0]]._p[1]) + InitY, InitZ});
            //                Coor.push_back({static_cast<double>(pointVec[seg._v[1]]._p[0]) + InitX,
            //                                static_cast<double>(pointVec[seg._v[1]]._p[1]) + InitY, InitZ});
            //                nlohmann::json LineObject;
            //                LineObject["type"] = "Feature";
            //                LineObject["properties"] = {{"LineID", TriID}};
            //                LineObject["geometry"] = {{"type",        "LineString"},
            //                                            {"coordinates", Coor}};
            //                JsonFile["features"].push_back(LineObject);
            //                //        OutputStream << seg._v[0] << " " << seg._v[1] << std::endl;
            //                ++TriID;
            //            }
            //
            //            OutputStream << JsonFile << std::endl;
            //            OutputStream.close();
        }
        else
        {
            std::cerr << _outMeshFilename << " is not a valid path!" << std::endl;
        }
    }
}

double TriangulationHandler::getTriNormal(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3)
{
    Eigen::Vector3d pt1pt2(pt2.x - pt1.x, pt2.y - pt1.y, pt2.z - pt1.z);
    Eigen::Vector3d pt2pt3(pt3.x - pt2.x, pt3.y - pt2.y, pt3.z - pt2.z);
    auto            Normal = pt1pt2.cross(pt2pt3);
    auto            Norm   = Normal(2) > 0 ? Normal : -Normal;
    //    double          c0     = pt1pt2(1) * pt2pt3(2) - pt1pt2(2) * pt2pt3(1);
    //    double          c1     = pt1pt2(2) * pt2pt3(0) - pt1pt2(0) * pt2pt3(2);
    //    double          c2     = pt1pt2(0) * pt2pt3(1) - pt1pt2(1) * pt2pt3(0);
    //    std::cout << std::setprecision(10);
    //    std::cout << pt1pt2(0) << " " << pt1pt2(1) << " " << pt1pt2(2) << std::endl;
    //    std::cout << pt2pt3(0) << " " << pt2pt3(1) << " " << pt2pt3(2) << std::endl;
    //    std::cout << Norm(0) << " " << Norm(1) << " " << Norm(2) << std::endl;
    //    std::cout << c0 << " " << c1 << " " << c2 << std::endl;
    Norm.normalize();

    if (Norm(0) != 0)
    {
        int x = 0;
    }
    Eigen::Vector3d Upward(0, 0, 1);
    return std::acos(Norm.dot(Upward)) * 180 / M_PI;
}

bool TriangulationHandler::hasValidEdge(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3)
{
    return (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y) < 1 &&
           (pt3.x - pt2.x) * (pt3.x - pt2.x) + (pt3.y - pt2.y) * (pt3.y - pt2.y) < 1 &&
           (pt1.x - pt3.x) * (pt1.x - pt3.x) + (pt1.y - pt3.y) * (pt1.y - pt3.y) < 1;
}