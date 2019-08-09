#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <vector>
#include <queue>
#include <functional>
#include <algorithm>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


int main(void)
{

    struct vertex
    {
        // cv::Point2f position;
        std::vector <int> adjacencyLists;
        std::vector <double> weights;
        double feasibleLabel;

        vertex()
        {
            adjacencyLists.reserve(100);
            weights.reserve(100);
        };
    };

    // std::vector < std::vector <int> > adjacencyLists;
    // std::vector <double> weights;
    
    
    //-----------
    // Load data
    //-----------
    
    //cv::Mat imageBGR(cv::Size(1200, 800), CV_8UC4, cv::Scalar(64, 64, 64));
    cv::Mat imageBGR = cv::imread("../matching.png", cv::IMREAD_UNCHANGED);

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
    
    cv::FileStorage fs("../matching.yml", cv::FileStorage::READ);


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
   
 

    // For each projected model point, try to assign best matching image point in local neighborhood
    for ( std::size_t i = 0; i < modelPoints.size(); i++ )
    {
        cv::Mat query = cv::Mat( modelPoints[ i ] ).reshape( 0, 1 ); // Reminder: channels, rows, O(1)
    
        // Radius search (approximate nearest neighbor)
        int numNeighbors = m_flannIndex->radiusSearch( query, indices, dists, searchRadius, searchParams ); // New interface
        std::cout << "Matching model point " << i<< ": numNeighbors: " << numNeighbors << std::endl;

        cv::Mat descriptorModel = modelDescriptors.row( i );
        vertex temp;
        
        if ( numNeighbors > 0 )
        {
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
            }
            A.push_back(temp);
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
 
