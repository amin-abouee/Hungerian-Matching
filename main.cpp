#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <vector>
#include <queue>
#include <functional>
#include <algorithm>
#include <queue>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct vertex
{
    // cv::Point2f position;
    std::vector <int> adjacencyLists;
    std::vector <double> weights;
    double label;
    bool free;
    int matchedIdx;

    vertex()
    {
        adjacencyLists.reserve(100);
        weights.reserve(100);
        label = 0.0;
        free = true;
        matchedIdx = -1;
    };
};


std::set <std::pair <int, int>  >findBFSPatches ( std::vector<vertex>& A, std::vector<vertex>&B, int u, int y, std::map<int, int>& mapImageIndexToVectorIdx)
{
    std::vector <int> parentsA (A.size(), -1);
    std::vector <int> parentsB (B.size(), -1);

    std::queue<int> Q;
    Q.push(u);
    parentsA[u] = u;

    while(!Q.empty())
    {
        int node = Q.front();
        Q.pop();
        for(int i(0); i < A[node].adjacencyLists.size(); i++)
        {
            int b = A[node].adjacencyLists[i];
            int vecBidx = mapImageIndexToVectorIdx[b];
            if (b == y)
            {
                parentsB[vecBidx] = node;
                while(!Q.empty())
                    Q.pop();
                break;
            }
            if (A[node].label + B[vecBidx].label == A[node].weights[i] && parentsB[vecBidx] == -1)
            {
                parentsB[vecBidx] = node;
                for(int j(0); j < B[vecBidx].adjacencyLists.size(); j++)
                {
                    int a = B[vecBidx].adjacencyLists[j];
                    if (B[vecBidx].label + A[a].label == B[vecBidx].weights[j] && parentsA[a] == -1)
                    {
                        Q.push(a);
                        parentsA[a] = b;
                    }
                }
            }
        }
    }

    std::set< std::pair < int, int> > M;
    int idb = y;
    int vecBidx = mapImageIndexToVectorIdx[idb];
    int ida = parentsB[vecBidx];
    M.insert(std::make_pair(ida, idb));
    A[ida].free = false;
    B[vecBidx].free = false;
    while(parentsA[ida] != ida)
    {
        idb = parentsA[ida];
        M.insert(std::make_pair(ida, idb));
        vecBidx = mapImageIndexToVectorIdx[idb];
        ida = parentsB[vecBidx];
        M.insert(std::make_pair(ida, idb));
    }
    return M;
}

int main(void)
{

    // std::vector < std::vector <int> > adjacencyLists;
    // std::vector <double> weights;
    
    
    //-----------
    // Load data
    //-----------
    
    //cv::Mat imageBGR(cv::Size(1200, 800), CV_8UC4, cv::Scalar(64, 64, 64));
    cv::Mat imageBGR = cv::imread("/home/amin/Workspace/cplusplus/graph_matching/matching.png", cv::IMREAD_UNCHANGED);

    cv::Mat imagePointsMat;
    cv::Mat modelPointsMat;
    //std::vector<cv::KeyPoint> imageKeyPoints;
    //std::vector<cv::KeyPoint> modelKeyPoints;
    cv::Mat imageDescriptors;
    cv::Mat modelDescriptors;
    std::vector< cv::Point2f > imagePoints; // Image points
    std::vector< cv::Point2f > modelPoints; // Projected model points

    std::vector <vertex> A;
    std::vector <vertex> B;
    
    std::vector<float> imageAngles; // Feature orientation in Radians
    std::vector<float> modelAngles;
    
    cv::FileStorage fs("/home/amin/Workspace/cplusplus/graph_matching/matching.yml", cv::FileStorage::READ);


    fs["imagePoints"] >> imagePointsMat;
    imagePointsMat.convertTo( imagePointsMat, CV_32FC1 );
    if ( imagePointsMat.isContinuous() )
    {
        imagePoints.assign( (cv::Point2f*)imagePointsMat.datastart, (cv::Point2f*)imagePointsMat.dataend );
    }

    fs["modelPoints"] >> modelPointsMat;
        modelPointsMat.convertTo( modelPointsMat, CV_32FC1 );
    if ( modelPointsMat.isContinuous() )
    {
        modelPoints.assign( (cv::Point2f*)modelPointsMat.datastart, (cv::Point2f*)modelPointsMat.dataend );
    }

    //fs["imageKeyPoints"] >> imageKeyPoints;
    //fs["modelKeyPoints"] >> modelKeyPoints;
    
    // Extract image coordinates of features from keypoints
    //cv::KeyPoint::convert(imageKeyPoints, imagePoints);
    //cv::KeyPoint::convert(modelKeyPoints, modelPoints);
    
    fs["imageDescriptors"] >> imageDescriptors;
    fs["modelDescriptors"] >> modelDescriptors;
    
    fs["imageAngles"] >> imageAngles;
    fs["modelAngles"] >> modelAngles;

    fs.release();
    
   
    std::cout << "imageKeyPoints.size(): " << imagePoints.size() << std::endl;
    std::cout << "modelKeyPoints.size(): " << modelPoints.size() << std::endl;
    
    

    //-------------
    // FLANN index
    //-------------
    cv::Mat m_pointsMat = cv::Mat(imagePoints).reshape(1); // Matrix has to be of size NxD (rows, cols) single channel and type CV_32FC1
    
    // Construct a nearest neighbor search index for current image
    cv::Ptr< cvflann::KDTreeSingleIndexParams > m_indexParams = new cvflann::KDTreeSingleIndexParams(4); // Approximate KdTree search consisting of N parallel random trees

    // Create the FLANN Index
    using flannDistance_t = cvflann::L2< float >;
    cv::Ptr< cv::flann::GenericIndex< flannDistance_t > > m_flannIndex = new cv::flann::GenericIndex< flannDistance_t > (m_pointsMat, *m_indexParams);
    


    //---------------------
    // FLANN radius search
    //---------------------
    float verificationRadius = 10.0;
    float searchRadius = std::pow(verificationRadius, 2); // Distance in image space [px]. ATTENTION: Depends on used norm, e.g. 5 px with cvflann::L1 are equivalent to 5^2 = 25 px with cvflann::L2
    int checks = 32;  //The number of times the tree(s) in the index should be recursively traversed
    float eps = 0.0; // Search for eps-approximate neighbors (whatever that means...)
    bool sorted = true; // Only used for radius search, TODO: is false faster?
    cvflann::SearchParams searchParams(checks, eps, sorted);  // New interface
    
    //double threshDistance = 100; // Maximum Hamming distance
    int maxResults = 10; // Restrict radius search
    cv::Mat indices( 1, maxResults, CV_32SC1 ); // Old interface: assign neither type nor size here (segfaults otherwise)!
    cv::Mat dists( 1, maxResults, CV_32FC1 ); // Old interface: assign neither type nor size here (segfaults otherwise)!

    std::map<int, int> mapImageIndexToVectorIdx;
   
 
    // For each projected model point, try to assign best matching image point in local neighborhood
    for ( std::size_t i = 0; i < modelPoints.size(); i++ )
    {
        cv::Mat query = cv::Mat( modelPoints[ i ] ).reshape( 0, 1 ); // Reminder: channels, rows, O(1)
    
        // Radius search (approximate nearest neighbor)
        int numNeighbors = m_flannIndex->radiusSearch( query, indices, dists, searchRadius, searchParams ); // New interface
        std::cout << "Matching model point " << i<< ": numNeighbors: " << numNeighbors << std::endl;

        cv::Mat descriptorModel = modelDescriptors.row( i );
        
        if ( numNeighbors > 0 )
        {
            vertex temp;
            // vertex tempB;
            // Compute distance
            for ( int j = 0; j < std::min( numNeighbors, maxResults ); j++ )
            {
                int idx = indices.at< int >( 0, j );
                cv::Mat descriptorImage = imageDescriptors.row( idx );
                double distDescriptor = cv::norm(descriptorModel, descriptorImage, cv::NORM_HAMMING);
                double distEuclidean = cv::norm( modelPoints[i] - imagePoints[idx] );
                double distAngle = std::pow( std::cos( modelAngles[i] - imageAngles[idx] ), 2);
                
                double allinone = (1.0 - distDescriptor/255.0) * (1.0 - distEuclidean/verificationRadius) * distAngle;
                std::cout << "   Neighbor " << j << ": " << "Weight: " << allinone <<  ", Descriptor: " << distDescriptor << ", Euclidean: " << distEuclidean << ", Angle: " << distAngle << std::endl;
                temp.adjacencyLists.push_back(idx);
                temp.weights.push_back(allinone);

                //  -- initial edge labeling --
                temp.label = std::max(temp.label, allinone);

                //  -- initial B vector, the other side of bipartite graph
                auto it = mapImageIndexToVectorIdx.find(idx);
                if (it != mapImageIndexToVectorIdx.cend())
                {
                    int imgIdx = it->second;
                    B[imgIdx].adjacencyLists.push_back(i);
                    B[imgIdx].weights.push_back(allinone);
                }
                else
                {
                    vertex tempB;
                    tempB.adjacencyLists.push_back(i);
                    tempB.weights.push_back(allinone);
                    B.push_back(tempB);
                    mapImageIndexToVectorIdx[idx] = B.size() - 1;
                }
            }
            A.push_back(temp);
        }
    }


    {
        A.clear();
        B.clear();
        mapImageIndexToVectorIdx.clear();

        vertex a1;
        a1.adjacencyLists.push_back(0);
        a1.weights.push_back(2);
        a1.adjacencyLists.push_back(1);
        a1.weights.push_back(3);
        a1.adjacencyLists.push_back(2);
        a1.weights.push_back(2);
        a1.label = 3;
        A.push_back(a1);

        vertex a2;
        a2.adjacencyLists.push_back(0);
        a2.weights.push_back(3);
        a2.adjacencyLists.push_back(1);
        a2.weights.push_back(4);
        a2.adjacencyLists.push_back(2);
        a2.weights.push_back(5);
        a2.label = 5;
        A.push_back(a2);

        vertex a3;
        a3.adjacencyLists.push_back(1);
        a3.weights.push_back(1);
        a3.adjacencyLists.push_back(2);
        a3.weights.push_back(3);
        a3.adjacencyLists.push_back(3);
        a3.weights.push_back(4);
        a3.label = 4;
        A.push_back(a3);

        vertex a4;
        a4.adjacencyLists.push_back(2);
        a4.weights.push_back(2);
        a4.adjacencyLists.push_back(3);
        a4.weights.push_back(4);
        a4.label = 4;
        A.push_back(a4);

        vertex b1;
        b1.adjacencyLists.push_back(0);
        b1.weights.push_back(2);
        b1.adjacencyLists.push_back(1);
        b1.weights.push_back(3);
        B.push_back(b1);

        vertex b2;
        b2.adjacencyLists.push_back(0);
        b2.weights.push_back(3);
        b2.adjacencyLists.push_back(1);
        b2.weights.push_back(4);
        b2.adjacencyLists.push_back(2);
        b2.weights.push_back(1);
        B.push_back(b2);

        vertex b3;
        b3.adjacencyLists.push_back(0);
        b3.weights.push_back(2);
        b3.adjacencyLists.push_back(1);
        b3.weights.push_back(3);
        b3.adjacencyLists.push_back(3);
        b3.weights.push_back(2);
        B.push_back(b3);

        vertex b4;
        b4.adjacencyLists.push_back(2);
        b4.weights.push_back(4);
        b4.adjacencyLists.push_back(3);
        b4.weights.push_back(4);
        B.push_back(b4);

        mapImageIndexToVectorIdx[0] = 0;
        mapImageIndexToVectorIdx[1] = 1;
        mapImageIndexToVectorIdx[2] = 2;
        mapImageIndexToVectorIdx[3] = 3;
    }

    // double check the connectivity between model idx and image idx, image idx maps to vector idx
    for(const auto& p : mapImageIndexToVectorIdx)
        std::cout << "connect: " << p.first << " , " << p.second << std::endl;

    int cnt = 0;
    for(const auto& v : A)
    {
        std::cout << "model idx: " << cnt << ", label: " << v.label << std::endl; 
        for(const auto i : v.adjacencyLists)
        {
            std::cout << i << " (" << mapImageIndexToVectorIdx[i] << ")  ";
        }
        std::cout << std::endl;
        cnt++;
    }

    bool initSets = true;
    std::set<int> S;
    std::set<int> T;
    std::set<int> NS;
    std::set <std::pair<int , int> > M;

    while(true)
    {
        int u = -1;
        // initialization of S and T, if necessary
        if (initSets == true)
        {
            for(uint32_t i(0); i<A.size(); i++)
            {
                if(A[i].free == true)
                {
                    S.insert(i);
                    u = i;
                    break;
                }
            }
            T.clear();
            initSets = false;
        }

        // calculation of NS
        for(const auto s: S)
        {
            for(int i(0); i < A[s].adjacencyLists.size(); i++ )
            {
                int idx = A[s].adjacencyLists[i];
                int idxVecB = mapImageIndexToVectorIdx[idx];
                if (A[s].label + B[idxVecB].label == A[s].weights[i])
                {
                    NS.insert(idx);
                }
            }
        }
        if (NS == T)
        {
            double delta = 1.0;
            for(const auto& s: S)
            {
                for (int i(0); i<A[s].adjacencyLists.size(); i++)
                {
                    const int idx = A[s].adjacencyLists[i];
                    const int vecBIdx = mapImageIndexToVectorIdx[idx];
                    if (T.find(idx) != T.end())
                    {
                        delta = std::min(delta, A[s].label + B[vecBIdx].label - A[s].weights[idx]);
                    }
                }
            }
            for(const auto& s: S)
                A[s].label -= delta;

            for(const auto& t: T)
            {
                const int vecBIdx = mapImageIndexToVectorIdx[t];
                B[vecBIdx].label += delta;
            }
        }
        else
        {
            for(const auto& ns : NS)
            {
                const int vecBIdx = mapImageIndexToVectorIdx[ns];
                if (B[vecBIdx].free == true && T.find(vecBIdx) == T.end())
                {
                    std::set< std::pair < int, int> > currentPath = findBFSPatches (A, B, u, ns, mapImageIndexToVectorIdx);
                    for(const auto& p : currentPath)
                    {
                        auto it = M.find(p);
                        if(it != M.end())
                            M.erase(it);
                        else
                            M.insert(p);
                    }
                }
                else
                {
                    T.insert(ns);
                    S.insert(B[vecBIdx].matchedIdx);
                }
            }
        }
    }

    
    //------------------
    // Visualize points
    //------------------

    // Image points
    for ( std::size_t i = 0; i < imagePoints.size(); i++ )
    {
        cv::circle( imageBGR, imagePoints[i], 6, cv::Scalar(204, 104, 0), 1, 8 );
    }
       
    // Projected model points
    for ( std::size_t i = 0; i < modelPoints.size(); i++ )
    {
        cv::circle( imageBGR, modelPoints[i], verificationRadius, cv::Scalar(104, 0, 204), 1, 8 );
    }
    cv::imshow("Image", imageBGR);
    cv::waitKey(0);

    
    return EXIT_SUCCESS;
}
 
