#ifndef GPUFIT_HYPERBOLA_PS_CUH_INCLUDED
#define GPUFIT_HYPERBOLA_PS_CUH_INCLUDED

#include <iostream>
using namespace std;

/* Description of the calculate_hyperbola function
* ===================================================
*
* This function calculates the values of one-dimensional hyperbola model functions
* and their partial derivatives with respect to the model parameters. 
*
* This function makes use of the user information data to pass in the 
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* Note that if no user information is provided, the (X) coordinate of the 
* first data value is assumed to be (0.0).  In this case, for a fit size of 
* M data points, the (X) coordinates of the data are simply the corresponding 
* array index values of the data array, starting from zero.
*
* There are three possibilities regarding the X values:
*
*   No X values provided: 
*
*       If no user information is provided, the (X) coordinate of the 
*       first data value is assumed to be (0.0).  In this case, for a 
*       fit size of M data points, the (X) coordinates of the data are 
*       simply the corresponding array index values of the data array, 
*       starting from zero.
*
*   X values provided for one fit:
*
*       If the user_info array contains the X values for one fit, then 
*       the same X values will be used for all fits.  In this case, the 
*       size of the user_info array (in bytes) must equal 
*       sizeof(REAL) * n_points.
*
*   Unique X values provided for all fits:
*
*       In this case, the user_info array must contain X values for each
*       fit in the dataset.  In this case, the size of the user_info array 
*       (in bytes) must equal sizeof(REAL) * n_points * nfits.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: source x-coordinate
*             p[1]: source y-coordinate
*             p[2]: source z-coordinate
*             p[3]: intecept term t0
*             p[4]: moveout velocity
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index.
*
* chunk_index: The chunk index. Used for indexing of user_info.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_hyperbola_ps function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_hyperbola_ps(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // indices

    REAL * user_info_float = (REAL*) user_info;

    int const chunk_begin = chunk_index * n_fits * n_points;
    int const fit_begin = fit_index * n_points * 4;

    REAL x = user_info_float[chunk_begin + fit_begin + (point_index * 4)];
    REAL y = user_info_float[chunk_begin + fit_begin + (point_index * 4) + 1];
    REAL z = user_info_float[chunk_begin + fit_begin + (point_index * 4) + 2];
    REAL phase = user_info_float[chunk_begin + fit_begin + (point_index * 4) + 3];

    REAL const * p = parameters;

    // value

    REAL dx = (x - p[0]); // (x - x0) term
    REAL dy = (y - p[1]); // (y - y0) term
    REAL dz = (z - p[2]); // (z - z0) term

    REAL t0 = p[3];                                              // t0
    // REAL v = p[4];                                               // v
    REAL r = sqrt((dx * dx) + (dy * dy) + (dz * dz));            // dist term

    // mask for P-phase and S-phases
    REAL vp = p[4] * phase;
    REAL vs = p[4] * (1 - phase) / 1.78; // <--- # NOTE: custom vpvs ratio input
    REAL v = vp+vs;
    
    value[point_index] =  (t0 * t0) + ((r * r) / (v * v)); // hyperbola forward model p-phase
    // value[point_index] =  (t0 * t0) + ((r * r) / (vs * vs)); // hyperbola forward model s-phase
    
    // // derivatives

    REAL * current_derivatives = derivative + point_index;
    current_derivatives[0 * n_points] = 2 * -dx / (v * v); // d f(x,y,z,t0,v)/dx * vp * p-phase-mask  
    current_derivatives[1 * n_points] = 2 * -dy / (v * v); // d f(x,y,z,t0,v)/dy * vp * p-phase-mask
    current_derivatives[2 * n_points] = 2 * -dz / (v * v); // d f(x,y,z,t0,v)/dz * vp * p-phase-mask
    current_derivatives[3 * n_points] = 2 * t0;              // d f(x,y,z,t0,v)/dt0
    current_derivatives[4 * n_points] = -2 * r * r / (v * v * v); // d f(x,y,z,t0,v)/dv * vp * p-phase-mask
}

#endif
