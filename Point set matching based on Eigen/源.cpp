#include <iostream>
#include <Eigen/Dense>
#include <Eigen/core>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <Eigen/SVD>


using MatrifD = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
void main()
{
	Eigen::Matrix<float, 4, 3> Pa;
	Pa << 1, 0, 0,
		0, 1, 0,
		0, 0, 0,
		0, 0, 2;
	std::cout << "Pa" << std::endl;
	std::cout << Pa << std::endl;

	Eigen::Matrix<float, 4, 3> Pb;
	Pb << 0, -1, 1,
		1, 0, 1,
		0, 0, 1,
		0, 0, 3;
	std::cout << "Pb" << std::endl;
	std::cout << Pb << std::endl;

	Eigen::Matrix<float, 3, 1> PaC;
	PaC << Pa.block<4, 1>(0, 0).mean(),
		Pa.block<4, 1>(0, 1).mean(),
		Pa.block<4, 1>(0, 2).mean();
	std::cout << "PaC" << std::endl;
	std::cout << PaC << std::endl;

	Eigen::Matrix<float, 3, 1> PbC;
	PbC << Pb.block<4, 1>(0, 0).mean(),
		Pb.block<4, 1>(0, 1).mean(),
		Pb.block<4, 1>(0, 2).mean();
	std::cout << "PbC" << std::endl;
	std::cout << PbC << std::endl;

	Eigen::Matrix<float, 3, 3> H;
	H = H.Constant(0);


	for (int i = 0; i < Pa.rows(); ++i)
	{
		H += (Pa.block<1, 3>(i, 0) - PaC.transpose()).transpose() * (Pb.block<1, 3>(i, 0) - PbC.transpose());
		//std::cout << "tmp" << std::endl;
		//std::cout << tmp << std::endl;

		//std::cout << "H" << std::endl;
		//std::cout << H << std::endl;
	}
	std::cout << "H" << std::endl;
	std::cout << H << std::endl;


	Eigen::JacobiSVD < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	auto U = svd.matrixU();
	auto V = svd.matrixV();
	auto A = svd.singularValues();


	std::cout << "U" << std::endl << U << std::endl;
	std::cout << "V" << std::endl << V << std::endl;
	std::cout << "A" << std::endl << A << std::endl;

	auto test = U * A.asDiagonal() * V.transpose();
	std::cout << "testUAV" << std::endl << test << std::endl;

	//Eigen::Matrix<float, 3, 3> R = V * U.transpose();
	auto R = V * U.transpose();
	std::cout << "R" << std::endl << R << std::endl;

	auto T = (-R) * PaC + PbC;
	std::cout << "T" << std::endl << T << std::endl;


	Eigen::Matrix<float, 4, 4> P;
	P = P.Constant(0);
	P.block<3, 3>(0, 0) = R;
	P.block<3, 1>(0, 3) = T;
	P(3, 3) = 1;
	std::cout << "P\n" << P << std::endl;


	Eigen::Matrix<float, 4, 4> Ma;
	Ma.block<3, 4>(0, 0) = Pa.block<4, 3>(0, 0).transpose();
	Ma.block<1, 4>(3, 0) = Eigen::Matrix<float, 1, 4>::Constant(1);
	auto AafterP = P * Ma;
	std::cout << "AafterP\n" << AafterP << std::endl;
	std::cout << "Pb\n" << Pb << std::endl;

	return;
}