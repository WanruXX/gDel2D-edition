#include "../inc/TriangulationHandler.h"
#include "../inc/json.h"
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
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

void TriangulationHandler::saveResultsToFile()
{
    std::ofstream CheckOutput(_outCheckFilename, std::ofstream::app);
    if (CheckOutput.is_open())
    {
        CheckOutput << "GridWidth," << GridSize << ",";
        CheckOutput << "PointNum," << _input.InputPointVec.size() << ",";
        CheckOutput << "Runs," << _runNum << ",";
        CheckOutput << "Input," << (PtCreatorPara._inFile ? PtCreatorPara._inFilename : DistStr[PtCreatorPara._dist])
                    << ",";
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

    //        auto GroundNorm = getGroundNorm();

    std::vector<std::vector<int>> point2TriMap(_input.InputPointVec.size());
    std::vector<bool>             ValidTri(_output.triVec.size(), false);
    for (int t = 0; t < _output.triVec.size(); ++t)
    {
        if (hasValidEdge(InputPointCloud[_output.triVec[t]._v[0]],
                         InputPointCloud[_output.triVec[t]._v[1]],
                         InputPointCloud[_output.triVec[t]._v[2]]))
        {
            ValidTri[t] = true;
            for (auto &v : _output.triVec[t]._v)
            {
                point2TriMap[v].push_back(t);
            }
        }
    }

    pcl::PointCloud<tPointXYZT1> TrjPC;
    pcl::io::loadPCDFile("/home/wxxs6p/Downloads/sbet_Jarna1st0_Run1.pcd", TrjPC);
    pcl::KdTreeFLANN<tPointXYZT1>     KDTree;
    pcl::Indices                      NearestPointIdx;
    std::vector<float>                NearestPointSquareDist;
    pcl::PointIndices::Ptr            Filtered(new pcl::PointIndices());
    pcl::PointCloud<tPointXYZT1>::Ptr InPCXYZT(new pcl::PointCloud<tPointXYZT1>);
    tPointXYZT1                       InPCPt{};
    for (auto &pt : InputPointCloud)
    {
        InPCPt.x = pt.x;
        InPCPt.y = pt.y;
        InPCXYZT->push_back(InPCPt);
    }
    KDTree.setInputCloud(InPCXYZT);
    double                       GroundNorm0  = 0;
    double                       GroundNorm1  = 0;
    double                       GroundNorm2  = 0;
    double                       GroundTriNum = 0;
    std::vector<Eigen::Vector3d> Norms;

    for (auto &TrjPt : TrjPC)
    {
        KDTree.nearestKSearch(TrjPt, 1, NearestPointIdx, NearestPointSquareDist);
        Point2 pt2{TrjPt.x, TrjPt.y};
        for (auto &t : point2TriMap[NearestPointIdx[0]])
        {
            auto tri = _output.triVec[t];
            if (checkInside(tri, pt2))
            {
                auto norm = getTriNormal(tri);
                GroundNorm0 += norm(0);
                GroundNorm1 += norm(1);
                GroundNorm2 += norm(2);
                GroundTriNum++;
                break;
            }
        }
    }
    Eigen::Vector3d GroundNorm{GroundNorm0 / GroundTriNum, GroundNorm1 / GroundTriNum, GroundNorm2 / GroundTriNum};
    GroundNorm.normalize();
    std::cout << "GroundNorm"
              << ": " << GroundNorm(0) << " " << GroundNorm(1) << " " << GroundNorm(2) << std::endl;

    for (auto &norm : Norms)
    {
        auto angle = std::acos(norm.dot(GroundNorm)) * 180 / M_PI;
        if (angle > 2.5)
        {
            std::cout << "get norm outlier " << angle << std::endl;
        }
    }
    std::vector<double> Upwardness(_output.triVec.size(), -180);
    int                 triid = 0;
    for (auto &tri : _output.triVec)
    {
        if (ValidTri[triid])
        {
            auto Norm =
                getTriNormal(InputPointCloud[tri._v[0]], InputPointCloud[tri._v[1]], InputPointCloud[tri._v[2]]);
            Upwardness[triid] = std::acos(Norm.dot(GroundNorm)) * 180 / M_PI;
        }
        ++triid;
    }
    for (std::size_t PtId = 0; PtId < point2TriMap.size(); ++PtId)
    {
        InputPointCloud[PtId].returnnumber = -180;
        for (auto &t : point2TriMap[PtId])
        {
            if (Upwardness[t] > InputPointCloud[PtId].returnnumber)
            {
                InputPointCloud[PtId].returnnumber = Upwardness[t];
            }
        }
    }
    pcl::io::savePCDFile("/home/wxxs6p/repo/gDel2D-edition/app/test1/R1_L250_ds_norm.pcd", InputPointCloud);

    //        std::ofstream MeshOutput(_outMeshFilename);
    //        std::string   MeshData;
    //        if (MeshOutput.is_open())
    //        {
    //
    //            MeshData =
    //                R"({"crs":{"properties":{"name":"urn:ogc:def:crs:EPSG::5556"},"type":"name"},"name":"left_4_edge_polygon","type":"FeatureCollection", "features":[)";
    //            MeshData += "\n";
    //            //            nlohmann::json JsonFile;
    //            //            JsonFile["type"]                      = "FeatureCollection";
    //            //            JsonFile["name"]                      = "left_4_edge_polygon";
    //            //            JsonFile["crs"]["type"]               = "name";
    //            //            JsonFile["crs"]["properties"]["name"] = "urn:ogc:def:crs:EPSG::5556";
    //            //            JsonFile["features"]                  = nlohmann::json::array();
    //            int triid = 0;
    //            for (auto &tri : _output.triVec)
    //            {
    //                if (hasValidEdge(InputPointCloud[tri._v[0]], InputPointCloud[tri._v[1]], InputPointCloud[tri._v[2]]))
    //                {
    //                    auto Norm = getTriNormal(InputPointCloud[tri._v[0]], InputPointCloud[tri._v[1]], InputPointCloud[tri._v[2]]);
    //                    MeshData +=
    //                        R"({ "type": "Feature", "properties": { "TriID": )" + std::to_string(triid) +
    //                        " , \"Upward_g\": " +
    ////                        std::to_string(getupwards(
    ////                            InputPointCloud[tri._v[0]], InputPointCloud[tri._v[1]], InputPointCloud[tri._v[2]])) +
    //                        std::to_string(std::acos(Norm.dot(GroundNorm)) * 180 / M_PI) +
    //                        R"( }, "geometry": { "type": "Polygon", "coordinates": [[[)" +
    //                        std::to_string(static_cast<double>(InputPointCloud[tri._v[0]].x) + InitX) + ", " +
    //                        std::to_string(static_cast<double>(InputPointCloud[tri._v[0]].y) + InitY) + "], [" +
    //                        std::to_string(static_cast<double>(InputPointCloud[tri._v[1]].x) + InitX) + ", " +
    //                        std::to_string(static_cast<double>(InputPointCloud[tri._v[1]].y) + InitY) + "], [" +
    //                        std::to_string(static_cast<double>(InputPointCloud[tri._v[2]].x) + InitX) + ", " +
    //                        std::to_string(static_cast<double>(InputPointCloud[tri._v[2]].y) + InitY) + "]]] } },\n";
    ////                    nlohmann::json Coor = nlohmann::json::array();
    //#ifndef DISABLE_PCL_INPUT
    ////                    Coor.push_back({static_cast<double>(InputPointCloud[tri._v[0]].x) + InitX,
    ////                                    static_cast<double>(InputPointCloud[tri._v[0]].y) + InitY,
    ////                                    static_cast<double>(InputPointCloud[tri._v[0]].z) + InitZ});
    ////                    Coor.push_back({static_cast<double>(InputPointCloud[tri._v[1]].x) + InitX,
    ////                                    static_cast<double>(InputPointCloud[tri._v[1]].y) + InitY,
    ////                                    static_cast<double>(InputPointCloud[tri._v[1]].z) + InitZ});
    ////                    Coor.push_back({static_cast<double>(InputPointCloud[tri._v[2]].x) + InitX,
    ////                                    static_cast<double>(InputPointCloud[tri._v[2]].y) + InitY,
    ////                                    static_cast<double>(InputPointCloud[tri._v[2]].z) + InitZ});
    //#else
    //                    Coor.push_back({static_cast<double>(_input.InputPointVec[tri._v[0]]._p[0]) + InitX,
    //                                    static_cast<double>(_input.InputPointVec[tri._v[0]]._p[1]) + InitY});
    //                    Coor.push_back({static_cast<double>(_input.InputPointVec[tri._v[1]]._p[0]) + InitX,
    //                                    static_cast<double>(_input.InputPointVec[tri._v[1]]._p[1]) + InitY});
    //                    Coor.push_back({static_cast<double>(_input.InputPointVec[tri._v[2]]._p[0]) + InitX,
    //                                    static_cast<double>(_input.InputPointVec[tri._v[2]]._p[1]) + InitY});
    //#endif
    //                    //                    nlohmann::json CoorWrapper = nlohmann::json::array();
    //                    //                    CoorWrapper.push_back(Coor);
    //                    //                    nlohmann::json TriangleObject;
    //                    //                    TriangleObject["type"]       = "Feature";
    //                    //                    TriangleObject["properties"] = {
    //                    //                        {"TriID", TriID},
    //                    //                        {"Upward",
    //                    //                         std::to_string(getTriNormal(
    //                    //                             InputPointCloud[tri._v[0]], InputPointCloud[tri._v[1]], InputPointCloud[tri._v[2]]))}};
    //                    //                    TriangleObject["geometry"] = {{"type", "Polygon"}, {"coordinates", CoorWrapper}};
    //                    //                    JsonFile["features"].push_back(TriangleObject);
    //                }
    //                ++triid;
    //            }
    //            //                MeshOutput << JsonFile << std::endl;
    //            MeshData.pop_back();
    //            MeshData.pop_back();
    //            MeshOutput << MeshData;
    //            MeshOutput << "]\n}" << std::endl;
    //            MeshOutput.close();
    //
    //            //            std::ofstream OutputStream(OutputFile);
    //            //            nlohmann::json JsonFile;
    //            //            JsonFile["type"] = "FeatureCollection";
    //            //            JsonFile["name"] = "left_4_edge";
    //            //            JsonFile["crs"]["type"] = "name";
    //            //            JsonFile["crs"]["properties"]["name"] = "urn:ogc:def:crs:EPSG::5556";
    //            //            JsonFile["features"] = nlohmann::json::array();
    //            //            int TriID = 0;
    //            //            for (auto &seg: segSet) {
    //            //                nlohmann::json Coor = nlohmann::json::array();
    //            //                Coor.push_back({static_cast<double>(pointVec[seg._v[0]]._p[0]) + InitX,
    //            //                                static_cast<double>(pointVec[seg._v[0]]._p[1]) + InitY, InitZ});
    //            //                Coor.push_back({static_cast<double>(pointVec[seg._v[1]]._p[0]) + InitX,
    //            //                                static_cast<double>(pointVec[seg._v[1]]._p[1]) + InitY, InitZ});
    //            //                nlohmann::json LineObject;
    //            //                LineObject["type"] = "Feature";
    //            //                LineObject["properties"] = {{"LineID", TriID}};
    //            //                LineObject["geometry"] = {{"type",        "LineString"},
    //            //                                            {"coordinates", Coor}};
    //            //                JsonFile["features"].push_back(LineObject);
    //            //                //        OutputStream << seg._v[0] << " " << seg._v[1] << std::endl;
    //            //                ++TriID;
    //            //            }
    //            //
    //            //            OutputStream << JsonFile << std::endl;
    //            //            OutputStream.close();
    //        }
    //        else
    //        {
    //            std::cerr << _outMeshFilename << " is not a valid path!" << std::endl;
    //        }
}

bool TriangulationHandler::checkInside(Tri &t, Point2 p) const
{
    // Create a point at infinity, y is same as point p
    line exline = {p, {p._p[0] + 320, p._p[1]}};
    int  count  = 0;
    for (auto i : TriSeg)
    {
        line side = {_input.InputPointVec[t._v[i[0]]], _input.InputPointVec[t._v[i[1]]]};
        if (isIntersect(side, exline))
        {

            // If side is intersects exline
            if (direction(side.p1, p, side.p2) == 0)
                return onLine(side, p);
            count++;
        }
    }
    // When count is odd
    return count & 1;
}

Eigen::Vector3d TriangulationHandler::getGroundNorm() const
{
    std::vector<std::vector<int>> point2TriMap(_input.InputPointVec.size());
    for (int t = 0; t < _output.triVec.size(); ++t)
    {
        if (hasValidEdge(InputPointCloud[_output.triVec[t]._v[0]],
                         InputPointCloud[_output.triVec[t]._v[1]],
                         InputPointCloud[_output.triVec[t]._v[2]]))
        {
            for (auto &v : _output.triVec[t]._v)
            {
                point2TriMap[v].push_back(t);
            }
        }
    }

    pcl::PointCloud<tPointXYZT1> TrjPC;
    pcl::io::loadPCDFile("/home/wxxs6p/Downloads/sbet_Jarna1st0_Run1.pcd", TrjPC);
    pcl::KdTreeFLANN<POINT_TYPE> KDTree;
    pcl::Indices                 NearestPointIdx;
    std::vector<float>           NearestPointSquareDist;
    pcl::PointIndices::Ptr       Filtered(new pcl::PointIndices());
    KDTree.setInputCloud(std::make_shared<const pcl::PointCloud<POINT_TYPE>>(InputPointCloud));
    double GroundNorm0  = 0;
    double GroundNorm1  = 0;
    double GroundNorm2  = 0;
    double GroundTriNum = 0;
    for (auto &TrjPt : TrjPC)
    {
        POINT_TYPE pt{};
        pt.x = TrjPt.x;
        pt.y = TrjPt.y;
        KDTree.nearestKSearch(pt, 1, NearestPointIdx, NearestPointSquareDist);
        Point2 pt2{TrjPt.x, TrjPt.y};
        for (auto &t : point2TriMap[NearestPointIdx[0]])
        {
            auto tri = _output.triVec[t];
            if (checkInside(tri, pt2))
            {
                auto norm = getTriNormal(tri);
                GroundNorm0 += norm(0);
                GroundNorm0 += norm(1);
                GroundNorm0 += norm(2);
                GroundTriNum++;
                break;
            }
        }
    }
    return {GroundNorm0 / GroundTriNum, GroundNorm1 / GroundTriNum, GroundNorm2 / GroundTriNum};
}

double TriangulationHandler::getupwards(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3)
{
    Eigen::Vector3d Upward(0, 0, 1);
    auto            Norm = getTriNormal(pt1, pt2, pt3);
    return std::acos(Norm.dot(Upward)) * 180 / M_PI;
}

Eigen::Vector3d
TriangulationHandler::getTriNormal(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3)
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
    return Norm;
}

Eigen::Vector3d TriangulationHandler::getTriNormal(const Tri &t) const
{
    auto pt1 = InputPointCloud[t._v[0]];
    auto pt2 = InputPointCloud[t._v[1]];
    auto pt3 = InputPointCloud[t._v[2]];
    return getTriNormal(pt1, pt2, pt3);
}

bool TriangulationHandler::hasValidEdge(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3)
{
    return (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y) < 1 &&
           (pt3.x - pt2.x) * (pt3.x - pt2.x) + (pt3.y - pt2.y) * (pt3.y - pt2.y) < 1 &&
           (pt1.x - pt3.x) * (pt1.x - pt3.x) + (pt1.y - pt3.y) * (pt1.y - pt3.y) < 1;
}