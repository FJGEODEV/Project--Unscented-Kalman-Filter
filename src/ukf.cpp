#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
	is_initialized_ = false;
	time_us_ = 0.0;
	n_x_ = 5;
  	n_aug_ = 7;
  	lambda_ = 3 - n_aug_;
  	weights_  = VectorXd(2*n_aug_+1);
  	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  	
	NIS_radar_ = 0.0;
  	NIS_laser_ = 0.0;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	if (!is_initialized_) {
	   
	    // init
	    x_ << 0, 0, 0, 0, 0;
	    P_ << 1, 0, 0, 0, 0,
	    	  0, 1, 0, 0, 0,
		      0, 0, 1, 0, 0,
		      0, 0, 0, 1, 0,
		      0, 0, 0, 0, 1;
	    time_us_ = meas_package.timestamp_;

	    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ ) {
	     
		float rho    = meas_package.raw_measurements_(0);
		float phi    = meas_package.raw_measurements_(1);
		float rhodot = meas_package.raw_measurements_(2);

		double vx = rhodot * cos(phi);
		double vy = rhodot * sin(phi);

		x_(0) = rho * cos(phi);
		x_(1) = rho * sin(phi);

		x_(2) = sqrt(vx*vx + vy*vy);
		x_(3) = 0;
		x_(4) = 0;

	    }
	    else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ ) {      
		x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
	    }

	is_initialized_ = true;
	return;
 	
      }

	 //For all types	   
	 double weight_0 = lambda_/(lambda_+n_aug_);
	 weights_(0) = weight_0;
	 int sum = 2*n_aug_+1;
	 //cout << "weight num:" << sum << endl;
	 for (int i=1; i<sum; i++) {  //2n+1 weights
	 	double weight = 0.5/(n_aug_+lambda_);
		weights_(i) = weight;
	        //cout << weights_(i) << endl;
	 } 

	 float dt;
	 dt = meas_package.timestamp_ - time_us_;
	 dt = dt / 1000000.0;
	 time_us_ = meas_package.timestamp_;
	 //cout << "dt =" << dt << endl;

	 Prediction(dt);

	 //cout << "Prediction is done!"  << endl;
	 if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
	   // Radar updates
	   // cout << "Ready to go in to Radar!"  << endl;
	      UpdateRadar(meas_package);
	   //cout << "OK, updated Radar!"  << endl;
	 }else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
	   // Laser updates
	   //cout << "Ready to go in to Lidar!"  << endl;
	     UpdateLidar(meas_package);
	   //cout << "OK, updated LIdar!"  << endl;
	 }else{
	     cout << "Project terminated because of missing condition." << endl;
	   }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

	//generate square root of P
	MatrixXd sqrtP = P_.llt().matrixL();
	
	// Augment sigma point - mainly from Udacity course
	VectorXd x_aug = VectorXd(n_aug_);
	// Create augment mean
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	//Now need to create real augment matrix
	MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P_; //Put matrix P to left corner
	P_aug(5, 5) = std_a_  * std_a_; // predefined
	P_aug(6, 6) = std_yawdd_ * std_yawdd_; // predefined

	//so sqrt of P_aug =
	MatrixXd sqrtPaug = P_aug.llt().matrixL();

	// based on equations to create augmented sigma points
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1); //
	Xsig_aug.col(0) = x_aug;
	for (int i=0;i<n_aug_;i++){
		Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * sqrtPaug.col(i);
		Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * sqrtPaug.col(i);
	}

	//now is step to predict sigma point
	for (int i=0; i<(2*n_aug_+1); i++){
		//assign Xsig_aug to different parameters for better reading
		double p_x      = Xsig_aug(0,i);
		double p_y      = Xsig_aug(1,i);
		double v        = Xsig_aug(2,i);
		double yaw      = Xsig_aug(3,i);
		double yawd     = Xsig_aug(4,i);
		double nu_a     = Xsig_aug(5,i);
		double nu_yawdd = Xsig_aug(6,i);

		double px_pred, py_pred;
		double v_pred, yaw_pred, yawd_pred;
		if (fabs(yawd)>0.001){
			px_pred = p_x + (v/yawd)*(sin(yaw+yawd*delta_t) - sin(yaw)) + 0.5*delta_t*delta_t*cos(yaw)*nu_a;
			py_pred = p_y + (v/yawd)*(cos(yaw) - cos(yaw+yawd*delta_t)) + 0.5*delta_t*delta_t*sin(yaw)*nu_a;
			v_pred = v + nu_a*delta_t;
			yaw_pred = yaw + yawd*delta_t+ 0.5*nu_yawdd*delta_t*delta_t;
			yawd_pred = yawd + nu_yawdd*delta_t;
		}else{
			px_pred = p_x + v*delta_t*cos(yaw) + 0.5*delta_t*delta_t*cos(yaw)*nu_a;
			py_pred = p_y + v*delta_t*sin(yaw) + 0.5*delta_t*delta_t*sin(yaw)*nu_a;
			v_pred = v + nu_a*delta_t;
			yaw_pred = yaw + 0.5*nu_yawdd*delta_t*delta_t;
			yawd_pred = yawd + nu_yawdd*delta_t;
		}

		//Assign predicted sigma to Xsig_pred_ matrix
		Xsig_pred_(0,i) = px_pred;
		Xsig_pred_(1,i) = py_pred;
		Xsig_pred_(2,i) = v_pred;
		Xsig_pred_(3,i) = yaw_pred;
		Xsig_pred_(4,i) = yawd_pred;
	}

	// now we have Xsig_pred, the following step will guide to predict Mean and Covariance
	// the main code are learned from Lesson 7 chapter 23.
	x_.fill(0);
	for (int i=0;i<2*n_aug_+1;i++){
	      x_ = x_ + weights_(i)*Xsig_pred_.col(i);
	}

	//predict state covariance matrix
	float pi = 3.1415926;
	P_.fill(0);  //**** forget to reset p_ to 0.0 make RMSE blow out!!
	for (int i=0;i<2*n_aug_+1;i++){
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//make sure angle is within -pi to +pi
		if ((x_diff(3) * 180.0 / pi) >  180.0) x_diff(3) -= 2*pi;
		if ((x_diff(3) * 180.0 / pi) < -180.0) x_diff(3) += 2*pi;
		P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

	float pi = 3.1415926;
	int n_z = 2;
	VectorXd z = meas_package.raw_measurements_;

	//MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, 2 * n_aug_ + 1);
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//transform sigma point to measurement space
	Zsig.fill(0);
	for (int i=0;i<2*n_aug_+1;i++){
		double p_x = Xsig_pred_(0,i);
		double p_y = Xsig_pred_(1,i);
		Zsig(0,i) = p_x;
		Zsig(1,i) = p_y;
	}

	//calculate mean of predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	// init
	z_pred.fill(0.0);
	//for (int i=0;i<2*n_aug_+1;i++){
	//	z_pred = z_pred + weights_(i)*Zsig.col(i);
	//}
	z_pred = Zsig * weights_;

	//calculate covariance matrix
	MatrixXd covM = MatrixXd(n_z, n_z);
	//init
	covM.fill(0.0);
	for (int i=0;i<2*n_aug_+1;i++){
		VectorXd z_diff = Zsig.col(i) - z_pred;
		covM = covM + weights_(i)*z_diff*z_diff.transpose();
	}

	// add noise
	MatrixXd noise = MatrixXd(n_z, n_z);
	noise << std_laspx_ *std_laspx_, 0, 0, std_laspy_*std_laspy_;

	covM = covM + noise;

	// for cross correlation. need to allocate another matrix
	MatrixXd xcorr = MatrixXd(n_x_, n_z);
	//init
	xcorr.fill(0.0);
	for (int i=0;i<2*n_aug_+1;i++){
		VectorXd z_diff = Zsig.col(i) - z_pred;
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		xcorr = xcorr + weights_(i)*x_diff*z_diff.transpose();
	}

	//gain
	MatrixXd gain = xcorr*covM.inverse();
	//diff
	VectorXd z_diff = z - z_pred;

	NIS_laser_ = z_diff.transpose()*covM.inverse()*z_diff;

	x_ = x_ + gain * z_diff;
	P_ = P_ - gain*covM*gain.transpose();

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

	float pi = 3.1415926;
	  int n_z = 3; //rho, phi, rho_dot
	  VectorXd z = meas_package.raw_measurements_;

	  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
	  Zsig.fill(0.0);
	 // cout << "so far so good!" << endl;
	 //transform sigma points into measurement spac
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

	    double p_x = Xsig_pred_(0,i);
	    double p_y = Xsig_pred_(1,i);
	    double v   = Xsig_pred_(2,i);
	    double yaw = Xsig_pred_(3,i);

	    double v1 = cos(yaw)*v;
	    double v2 = sin(yaw)*v;

	    //check for zeros XXXXXXXXXXXXXX
	    //    if (fabs(p_x) < 0.001) {
	    //     p_x = 0.001;
	    //    }
	    //    if (fabs(p_y) < 0.001) {
	    //     p_y = 0.001;
	    //    }

	    // measurement model
	    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
	    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
	    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
	  }
	 // cout << "Init Zsig" << endl;
	  //Following is basicly similar with Lidar code, should write in another subroutine for simplicity

	  //predicted measurements
	  VectorXd z_pred = VectorXd(n_z);
	  z_pred.fill(0.0);
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	  }

	  //z_pred = Zsig * weights_;
	  //cout << "Init z_pred" << endl;

	  //covariance matrix covM
	  MatrixXd covM = MatrixXd(n_z,n_z);
	  //init
	  covM.fill(0.0);
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
			VectorXd z_diff = Zsig.col(i) - z_pred;
			//check to see if angle is within -pi to +pi
			if ((z_diff(1) * 180.0 / pi) >  180.0) z_diff(1) -= 2*pi;
			if ((z_diff(1) * 180.0 / pi) < -180.0) z_diff(1) += 2*pi;

			covM = covM + weights_(i)*z_diff*z_diff.transpose();
		}

	  //cout << "Init covM" << endl;

		MatrixXd noiseR = MatrixXd(n_z, n_z);
		noiseR << std_radr_ *std_radr_,                       0, 0,
                                     0, std_radphi_*std_radphi_, 0,
                                     0,                       0, std_radrd_*std_radrd_;

		covM = covM + noiseR;
		//cout << "Init noise" << endl;

		// for cross correlation. need to allocate another matrix
		MatrixXd xcorr = MatrixXd(n_x_, n_z);
		//init
		xcorr.fill(0.0);
		for (int i=0;i<2*n_aug_+1;i++){
			VectorXd z_diff = Zsig.col(i) - z_pred;
			//check to see if angle is within -pi to +pi
			if ((z_diff(1) * 180.0 / pi) >  180.0) z_diff(1) -= 2*pi;
			if ((z_diff(1) * 180.0 / pi) < -180.0) z_diff(1) += 2*pi;
			VectorXd x_diff = Xsig_pred_.col(i) - x_;
			//check to see if angle is within -pi to +pi
			if ((x_diff(3) * 180.0 / pi) >  180.0) x_diff(3) -= 2*pi;
			if ((x_diff(3) * 180.0 / pi) < -180.0) x_diff(3) += 2*pi;
			xcorr = xcorr + weights_(i)*x_diff*z_diff.transpose();
		}

		//cout << "Init xcorr" << endl;

		//gain
		MatrixXd GainK = xcorr*covM.inverse();
		//diff
		VectorXd z_diff = z - z_pred;
		//cout << "set up gain and z_diff" << endl;
		//check to see if angle is within -pi to +pi
		 if ((z_diff(1) * 180.0 / pi) >  180.0) z_diff(1) -= 2*pi;
		 if ((z_diff(1) * 180.0 / pi) < -180.0) z_diff(1) += 2*pi;

		//cout << "ready to calculate NIS_radar" << endl;
		NIS_radar_ = z_diff.transpose() * covM.inverse()*z_diff;
		//cout << "calculate NIS_radar" << endl;

		x_ = x_ + GainK * z_diff;
		//cout << "x_ is updated" << endl;

		P_ = P_ - GainK*covM*GainK.transpose();

		//cout << "P_ is updated" << endl;

}
