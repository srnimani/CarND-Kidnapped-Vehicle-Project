/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    // Define the number of particles
    num_particles = 400;
    
    weights.resize(num_particles);
    particles.resize(num_particles);
    
    default_random_engine gen;
    
    // Define and set standard deviation for x, y and psi from the passed parameters
    double std_x        = std[0];
    double std_y        = std[1];
    double std_theta    = std[2];
    
    // Create a normal (Gaussian) distribution for x, y and psi.
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
    
    // Create and set particle values for all
    for (int i = 0; i < num_particles; ++i)
    {
        Particle particle;
        particle.id     = i;
        particle.x      = dist_x(gen);
        particle.y      = dist_y(gen);
        particle.theta  = dist_theta(gen);
        particle.weight = 1;
        
        // Add this particle to the particle filter
        particles[i]    = particle;
        weights[i]      = particle.weight;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    default_random_engine gen;
    
    // Set standard deviation for x, y and theta from the passed parameters
    double std_x       = std_pos[0];
    double std_y       = std_pos[1];
    double std_theta   = std_pos[2];
    
    // Update all partcile values
    for (int i = 0; i < num_particles; ++i)
    {
        // Using temp. varaiables to speed up run time
        double current_particle_x     = particles[i].x;
        double current_particle_y     = particles[i].y;
        double current_particle_theta = particles[i].theta;
        double yaw_rate_mult_delta_t  = yaw_rate * delta_t;
        
        if (fabs(yaw_rate) > 0.00001) // Check for divide by 0 error
        {
            // Update the position and add gaussian noise
            double velocity_by_yaw_rate     = velocity / yaw_rate;
            current_particle_x       += velocity_by_yaw_rate * (sin(current_particle_theta + yaw_rate_mult_delta_t) - sin(current_particle_theta));
            current_particle_y       += velocity_by_yaw_rate * (cos(current_particle_theta) - cos(current_particle_theta + yaw_rate_mult_delta_t));
            current_particle_theta   += yaw_rate_mult_delta_t;
        }
        else
        {
            current_particle_x      += velocity * cos(current_particle_theta) * delta_t;
            current_particle_y      += velocity * sin(current_particle_theta) * delta_t;
            current_particle_theta  += yaw_rate_mult_delta_t;
        }
        
        // Adding gaussian noise
        normal_distribution<double> dist_x(current_particle_x, std_x);
        normal_distribution<double> dist_y(current_particle_y, std_y);
        normal_distribution<double> dist_theta(current_particle_theta, std_theta);
        
        // Update the current particle in the array
        particles[i].x      = dist_x(gen);
        particles[i].y      = dist_y(gen);
        particles[i].theta  = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    
    for(auto predicted : predicted)
    {
        double smallest_distance = numeric_limits<double>::max(); // set upper limit to start with
        for(auto observation : observations)
        {
            // Find the distance between observations and the landmarks
            double current_distance = dist(observation.x, observation.y, predicted.x, predicted.y);
            int nearest_id = -1 ;
            
            if(current_distance < smallest_distance)
            {
                nearest_id = predicted.id;
                smallest_distance = current_distance;
            }
            observation.id = nearest_id;
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedisa.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
    
    // get the particle standard deviations for the landmarks
    double std_x        = std_landmark[0];
    double std_y        = std_landmark[1];
    double weights_sum  = 0;

    // For each particle:
    for(int i = 0; i < num_particles ; i++)
    {
        double weight = 1.0;
        
        for(int j = 0; j < observations.size(); j++) // For each observation
        {
            // Convert observations from Vehicle to Map space
            LandmarkObs current_obs = observations[j];
            LandmarkObs current_obs_post_transformation;
            
            // Transform the coordinates from vehicle to worldspace
            current_obs_post_transformation.x = (current_obs.x * cos(particles[i].theta)) - (current_obs.y * sin(particles[i].theta)) + particles[i].x;
            current_obs_post_transformation.y = (current_obs.x * sin(particles[i].theta)) + (current_obs.y * cos(particles[i].theta)) + particles[i].y;
            current_obs_post_transformation.id = current_obs.id;
            
            // Associate the closest measurement to a given landmark
            Map::single_landmark_s landmark;
            double smallest_dist = numeric_limits<double>::max();
            
            for(int k = 0; k < map_landmarks.landmark_list.size(); k++) // For each landmark
            {
                Map::single_landmark_s current_landmark = map_landmarks.landmark_list[k];
                double current_distance = dist(current_obs_post_transformation.x, current_obs_post_transformation.y, current_landmark.x_f, current_landmark.y_f);
                if(current_distance < smallest_dist)
                {
                    smallest_dist = current_distance;
                    landmark = current_landmark;
                }
            }
            
            // Update weights, define temp. variables to speed up processing
            double factor_x = current_obs_post_transformation.x - landmark.x_f;
            double factor_y = current_obs_post_transformation.y - landmark.y_f;
            
            weight *= 0.5 * exp(-0.5 * (pow(factor_x, 2)/pow(std_x, 2) + pow(factor_y, 2)/pow(std_y, 2)))/(M_PI * std_x * std_y) ;
        }
        
        weights_sum += weight;
        particles[i].weight = weight;
    }
    
    // Normalize the weights
    
    for (int i = 0; i < num_particles; i++)
    {
        particles[i].weight /= weights_sum;
        weights[i] = particles[i].weight;
    }
    
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    default_random_engine gen;
    
    // Use disctere distribution function to generate random integers in the interbval 0 to n, where the probability of each individual integer i is
    // defined as the weight of the ith integer divided by the sum of all n weights.
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    
    vector<Particle> resampled_particles;
    vector<double> resampled_weights;
    
    for (int i = 0; i < num_particles; i++)
    {
        resampled_particles.push_back(particles[distribution(gen)]);
        resampled_weights.push_back(particles[distribution(gen)].weight);
    }
    
    // Replace the original particles and weights with the new samples and weights
    particles   = resampled_particles;
    weights     = resampled_weights;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i)
    {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
